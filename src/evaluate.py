"""Evaluate baseline and improved fraud models and save comparison artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from config import (
    BASELINE_MODEL_PATH,
    CV_COMPARISON_TABLE_PATH,
    EVALUATION_INTERPRETATION_PATH,
    EVALUATION_METRICS_SUMMARY_PATH,
    FIGURES_DIR,
    IMPROVED_MODEL_PATH,
    METRICS_TABLE_PATH,
    MODEL_COMPARISON_SUMMARY_PATH,
    MODEL_COMPARISON_TABLE_PATH,
    PREPROCESSED_DATA_PATH,
    RANDOM_STATE,
    ROC_COMPARISON_FIGURE_PATH,
    RAW_DATA_PATH,
    TABLES_DIR,
    TARGET_COLUMN,
    VALIDATION_NOTES_PATH,
)
from data_preprocessing import (
    build_preprocessor,
    load_dataset_splits,
    run_preprocessing,
    select_numeric_features,
)
from train import build_baseline_model, build_improved_model
from utils import ensure_directories, set_seed


def load_test_data(data_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load preprocessed test arrays, creating them if missing."""
    if not PREPROCESSED_DATA_PATH.exists():
        run_preprocessing(data_path)
    arrays = np.load(PREPROCESSED_DATA_PATH)
    return arrays["X_test"], arrays["y_test"]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute key fraud-detection metrics for imbalanced data."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def save_confusion_matrix_plot(model_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> Path:
    """Save confusion matrix plot and return path."""
    model_tag = model_name.replace(" ", "_")
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, colorbar=False)
    ax.set_title(f"{model_name} - Confusion Matrix")
    fig.tight_layout()
    output_path = FIGURES_DIR / f"{model_tag}_confusion_matrix.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def get_feature_names(data_path: Path, n_features: int) -> list[str]:
    """Get feature names from dataset columns with a safe fallback."""
    if PREPROCESSED_DATA_PATH.exists():
        arrays = np.load(PREPROCESSED_DATA_PATH, allow_pickle=True)
        if "feature_names" in arrays:
            feature_names = [str(name) for name in arrays["feature_names"].tolist()]
            if len(feature_names) == n_features:
                return feature_names

    if data_path.exists() and data_path.is_file():
        header = pd.read_csv(data_path, nrows=1)
        feature_names = [
            col
            for col in header.columns
            if col != TARGET_COLUMN and not col.lower().startswith("unnamed:")
        ]
        if len(feature_names) == n_features:
            return feature_names
    return [f"feature_{i}" for i in range(n_features)]


def save_feature_importance_plot(
    model_name: str,
    model,
    feature_names: list[str],
    top_n: int = 15,
) -> tuple[Path | None, Path | None]:
    """Save feature importance figure/table when model supports it."""
    if not hasattr(model, "feature_importances_"):
        return None, None

    model_tag = model_name.replace(" ", "_")
    importances = np.asarray(model.feature_importances_)
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)
    top_df = importance_df.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top_df["feature"], top_df["importance"], color="#4c72b0")
    ax.set_title(f"{model_name} - Top Feature Importances")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()

    figure_path = FIGURES_DIR / f"{model_tag}_feature_importance.png"
    table_path = TABLES_DIR / f"{model_tag}_feature_importance.csv"
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    importance_df.to_csv(table_path, index=False)
    return figure_path, table_path


def evaluate_model(model_name: str, model, X_test: np.ndarray, y_test: np.ndarray) -> tuple[dict, np.ndarray]:
    """Evaluate one model on holdout test data and save confusion matrix."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_prob)
    confusion_path = save_confusion_matrix_plot(model_name, y_test, y_pred)
    row = {"model": model_name, **metrics, "confusion_matrix_path": str(confusion_path)}
    return row, y_prob


def save_metrics_summary_table(comparison_df: pd.DataFrame) -> Path:
    """Save a compact metrics summary table for all evaluated models."""
    metrics_summary = comparison_df[
        ["model", "accuracy", "precision", "recall", "f1", "roc_auc", "tn", "fp", "fn", "tp"]
    ].copy()
    metrics_summary.to_csv(EVALUATION_METRICS_SUMMARY_PATH, index=False)
    return EVALUATION_METRICS_SUMMARY_PATH


def save_roc_comparison_plot(y_true: np.ndarray, model_probs: dict[str, np.ndarray]) -> Path:
    """Save one ROC figure comparing all models."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for model_name, y_prob in model_probs.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_value = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC={auc_value:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_title("ROC Curve Comparison")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(ROC_COMPARISON_FIGURE_PATH, dpi=200)
    plt.close(fig)
    return ROC_COMPARISON_FIGURE_PATH


def save_evaluation_interpretation(comparison_df: pd.DataFrame) -> Path:
    """Save concise report-ready interpretation in academic English."""
    ranking = comparison_df.sort_values(
        by=["recall", "f1", "precision", "roc_auc"],
        ascending=False,
    ).reset_index(drop=True)
    best = ranking.iloc[0]
    other = ranking.iloc[1] if len(ranking) > 1 else None

    lines = [
        "# Evaluation Interpretation",
        "",
        (
            "In this imbalanced fraud detection task, model quality should be judged primarily "
            "by recall and F1-score rather than accuracy alone."
        ),
        (
            f"The best-performing model on the holdout test set is **{best['model']}** "
            f"(precision={best['precision']:.4f}, recall={best['recall']:.4f}, "
            f"F1={best['f1']:.4f}, ROC-AUC={best['roc_auc']:.4f})."
        ),
    ]

    if other is not None:
        lines.append(
            (
                f"Compared with **{other['model']}**, it offers a stronger fraud-detection tradeoff, "
                "capturing more fraudulent transactions (higher recall) while maintaining competitive "
                "precision. This balance is important because increasing recall can raise false positives."
            )
        )

    lines.append(
        "Overall, the preferred model should be selected based on operational tolerance for false alarms "
        "versus missed fraud cases, with priority typically given to minimizing missed fraud."
    )

    EVALUATION_INTERPRETATION_PATH.write_text("\n\n".join(lines), encoding="utf-8")
    return EVALUATION_INTERPRETATION_PATH


def run_stratified_cv(data_path: Path, n_splits: int) -> pd.DataFrame:
    """Run optional stratified k-fold CV on the training split only."""
    if n_splits < 2:
        raise ValueError("cv-folds must be at least 2 when cross-validation is enabled.")

    X_train_raw, _, y_train, _, _, _ = load_dataset_splits(data_path)
    X_train, _, feature_names = select_numeric_features(X_train_raw, X_train_raw)
    min_class_count = int(y_train.value_counts().min())
    if n_splits > min_class_count:
        raise ValueError(
            f"cv-folds={n_splits} is too large for the minority class in training data "
            f"(minimum class count={min_class_count})."
        )

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"precision": "precision", "recall": "recall", "f1": "f1", "roc_auc": "roc_auc"}
    model_builders = {
        "baseline_logistic_regression": build_baseline_model,
        "improved_random_forest": build_improved_model,
    }

    summary_rows = []
    fold_rows = []

    for model_name, model_builder in model_builders.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(feature_names)),
                ("model", model_builder()),
            ]
        )

        # Leakage guard: preprocessing is fit separately inside each fold's train partition.
        cv_results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            return_train_score=False,
        )

        row = {"model": model_name, "cv_folds": int(n_splits)}
        for metric_name in scoring:
            scores = cv_results[f"test_{metric_name}"]
            row[f"{metric_name}_mean"] = float(np.mean(scores))
            row[f"{metric_name}_std"] = float(np.std(scores, ddof=0))
            for fold_idx, score in enumerate(scores, start=1):
                fold_rows.append(
                    {
                        "model": model_name,
                        "fold": fold_idx,
                        "metric": metric_name,
                        "value": float(score),
                    }
                )
        summary_rows.append(row)

    cv_summary_df = pd.DataFrame(summary_rows).sort_values(
        by=["recall_mean", "f1_mean", "roc_auc_mean", "precision_mean"],
        ascending=False,
    )
    cv_summary_df.to_csv(CV_COMPARISON_TABLE_PATH, index=False)
    pd.DataFrame(fold_rows).to_csv(TABLES_DIR / "cv_fold_details.csv", index=False)
    return cv_summary_df


def save_comparison_summary(comparison_df: pd.DataFrame) -> None:
    """Write concise holdout summary with fraud-focused metric priority."""
    ranking = comparison_df.sort_values(
        by=["recall", "f1", "roc_auc", "precision"],
        ascending=False,
    ).reset_index(drop=True)
    best = ranking.iloc[0]
    runner_up = ranking.iloc[1] if len(ranking) > 1 else None

    lines = [
        "# Model Comparison Summary",
        "",
        "Fraud detection is class-imbalanced, so recall and F1 are prioritized over accuracy.",
        "",
        (
            f"Best holdout model: **{best['model']}** "
            f"(Recall={best['recall']:.4f}, F1={best['f1']:.4f}, ROC-AUC={best['roc_auc']:.4f})."
        ),
    ]
    if runner_up is not None:
        lines.append(
            (
                f"Compared to **{runner_up['model']}**, the selected model is preferred because it "
                "captures more fraud cases (higher recall) while keeping a better precision-recall balance "
                "(higher F1)."
            )
        )
    lines.append("")
    lines.append(
        "Recommendation: use the selected model for fraud screening when missing fraud is costlier "
        "than reviewing additional flagged transactions."
    )
    MODEL_COMPARISON_SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def save_validation_notes(run_cv: bool, cv_folds: int) -> None:
    """Save short markdown explanation of leakage prevention strategy."""
    lines = [
        "# Validation and Leakage Prevention Notes",
        "",
        "1. **Stratified split**",
        "- Train/test data is created with a stratified split so both sets preserve the fraud class ratio.",
        "",
        "2. **Holdout preprocessing safety**",
        "- The preprocessor is fit on training data only, then applied to test data.",
        "- Test labels and test feature statistics are never used during model fitting.",
        "",
        "3. **Cross-validation preprocessing safety**",
        (
            "- During CV, preprocessing and model are wrapped in one sklearn Pipeline, so each fold "
            "fits preprocessing only on that fold's training partition."
        ),
        "- This avoids leakage from validation folds into preprocessing parameters.",
    ]

    if run_cv:
        lines.extend(
            [
                "",
                f"4. **CV setting used in this run**",
                f"- StratifiedKFold with `{cv_folds}` folds, shuffle=True, random_state={RANDOM_STATE}.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "4. **CV setting used in this run**",
                "- Cross-validation was not requested in this run (`--run-cv` not provided).",
            ]
        )

    VALIDATION_NOTES_PATH.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Evaluate and compare fraud detection models.")
    parser.add_argument(
        "--data-path",
        default=str(RAW_DATA_PATH),
        help="Path to data/raw directory, or to a CSV file.",
    )
    parser.add_argument(
        "--baseline-model-path",
        default=str(BASELINE_MODEL_PATH),
        help="Path to baseline model file",
    )
    parser.add_argument(
        "--improved-model-path",
        default=str(IMPROVED_MODEL_PATH),
        help="Path to improved model file",
    )
    parser.add_argument(
        "--run-cv",
        action="store_true",
        help="Run optional stratified k-fold cross-validation on training split.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of folds for StratifiedKFold when --run-cv is enabled.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for model comparison."""
    args = parse_args()
    set_seed(RANDOM_STATE)
    ensure_directories(FIGURES_DIR, TABLES_DIR)

    baseline_model_path = Path(args.baseline_model_path)
    improved_model_path = Path(args.improved_model_path)
    if not baseline_model_path.exists() or not improved_model_path.exists():
        raise FileNotFoundError(
            "Model files are missing. Run `python src/train.py` first to train baseline and improved models."
        )

    data_path = Path(args.data_path)
    X_test, y_test = load_test_data(data_path)
    models = {
        "baseline_logistic_regression": joblib.load(baseline_model_path),
        "improved_random_forest": joblib.load(improved_model_path),
    }

    rows = []
    roc_probabilities: dict[str, np.ndarray] = {}
    for model_name, model in models.items():
        row, y_prob = evaluate_model(model_name, model, X_test, y_test)
        rows.append(row)
        roc_probabilities[model_name] = y_prob

    comparison_df = pd.DataFrame(rows).sort_values(
        by=["recall", "f1", "roc_auc", "precision"],
        ascending=False,
    )
    comparison_df.to_csv(MODEL_COMPARISON_TABLE_PATH, index=False)
    comparison_df.to_csv(METRICS_TABLE_PATH, index=False)
    metrics_summary_path = save_metrics_summary_table(comparison_df)
    roc_comparison_path = save_roc_comparison_plot(y_test, roc_probabilities)

    feature_names = get_feature_names(data_path, X_test.shape[1])
    importance_figure_path, importance_table_path = save_feature_importance_plot(
        "improved_random_forest",
        models["improved_random_forest"],
        feature_names=feature_names,
    )
    save_comparison_summary(comparison_df)
    interpretation_path = save_evaluation_interpretation(comparison_df)

    if args.run_cv:
        cv_summary_df = run_stratified_cv(data_path, n_splits=args.cv_folds)
        print(f"Saved CV comparison table: {CV_COMPARISON_TABLE_PATH}")
        print(f"Best CV model by recall/F1: {cv_summary_df.iloc[0]['model']}")

    save_validation_notes(run_cv=args.run_cv, cv_folds=args.cv_folds)

    print(f"Saved holdout comparison table: {MODEL_COMPARISON_TABLE_PATH}")
    print(f"Saved metrics table: {METRICS_TABLE_PATH}")
    print(f"Saved metrics summary table: {metrics_summary_path}")
    print(f"Saved ROC comparison figure: {roc_comparison_path}")
    if importance_figure_path is not None and importance_table_path is not None:
        print(f"Saved feature importance figure: {importance_figure_path}")
        print(f"Saved feature importance table: {importance_table_path}")
    print(f"Saved report summary: {MODEL_COMPARISON_SUMMARY_PATH}")
    print(f"Saved evaluation interpretation: {interpretation_path}")
    print(f"Saved validation notes: {VALIDATION_NOTES_PATH}")
    print(f"Saved confusion matrices in: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
