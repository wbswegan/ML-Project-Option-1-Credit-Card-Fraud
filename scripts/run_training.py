from pathlib import Path
import argparse
import sys

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from fraud_detection.config import (  # noqa: E402
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    RANDOM_STATE,
    RAW_DATA_PATH,
    REPORTS_DIR,
    TABLES_DIR,
    TARGET_COLUMN,
)
from fraud_detection.data import load_dataset, split_data  # noqa: E402
from fraud_detection.evaluation import (  # noqa: E402
    compute_metrics,
    plot_confusion,
    plot_precision_recall,
    plot_probability_histogram,
    plot_roc,
    save_classification_report,
)
from fraud_detection.models import build_baseline_model, build_improved_model  # noqa: E402
from fraud_detection.preprocessing import build_preprocessor  # noqa: E402
from fraud_detection.reporting import save_markdown_summary  # noqa: E402
from fraud_detection.utils import ensure_directories, save_json, set_global_seed  # noqa: E402


def evaluate_and_save(model_name: str, fitted_pipeline, X_test, y_test):
    """Evaluate one fitted model and save all artifacts."""
    y_pred = fitted_pipeline.predict(X_test)
    y_prob = fitted_pipeline.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_prob)

    model_tag = model_name.replace(" ", "_")
    save_json(metrics, METRICS_DIR / f"{model_tag}_metrics.json")
    save_classification_report(y_test, y_pred, TABLES_DIR / f"{model_tag}_classification_report.csv")
    plot_confusion(y_test, y_pred, FIGURES_DIR / f"{model_tag}_confusion_matrix.png", model_name)
    plot_roc(y_test, y_prob, FIGURES_DIR / f"{model_tag}_roc_curve.png", model_name)
    plot_precision_recall(
        y_test,
        y_prob,
        FIGURES_DIR / f"{model_tag}_precision_recall_curve.png",
        model_name,
    )
    plot_probability_histogram(
        y_test,
        y_prob,
        FIGURES_DIR / f"{model_tag}_probability_histogram.png",
        model_name,
    )

    return metrics


def train_baseline(X_train, y_train, X_test, y_test, feature_names):
    baseline_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(feature_names=feature_names, scale_numeric=True)),
            ("model", build_baseline_model()),
        ]
    )
    baseline_pipeline.fit(X_train, y_train)

    metrics = evaluate_and_save(
        model_name="baseline_logistic_regression",
        fitted_pipeline=baseline_pipeline,
        X_test=X_test,
        y_test=y_test,
    )
    joblib.dump(baseline_pipeline, MODELS_DIR / "baseline_logistic_regression.joblib")
    return metrics


def train_improved(X_train, y_train, X_test, y_test, feature_names, model_choice):
    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())
    scale_pos_weight = negative_count / max(positive_count, 1)

    chosen_name, chosen_model = build_improved_model(
        model_choice=model_choice,
        scale_pos_weight=scale_pos_weight,
    )
    improved_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(feature_names=feature_names, scale_numeric=False)),
            ("model", chosen_model),
        ]
    )
    improved_pipeline.fit(X_train, y_train)

    metrics = evaluate_and_save(
        model_name=f"improved_{chosen_name}",
        fitted_pipeline=improved_pipeline,
        X_test=X_test,
        y_test=y_test,
    )
    joblib.dump(improved_pipeline, MODELS_DIR / f"improved_{chosen_name}.joblib")
    return chosen_name, metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train fraud detection models.")
    parser.add_argument(
        "--data-path",
        default=str(RAW_DATA_PATH),
        help="Path to creditcard.csv",
    )
    parser.add_argument(
        "--improved-model",
        default="auto",
        choices=["auto", "xgboost", "random_forest"],
        help="Select improved model. 'auto' tries xgboost, then falls back to random_forest.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(RANDOM_STATE)
    ensure_directories([FIGURES_DIR, TABLES_DIR, MODELS_DIR, METRICS_DIR, REPORTS_DIR])

    df = load_dataset(Path(args.data_path))
    X_train, X_test, y_train, y_test = split_data(df, target_column=TARGET_COLUMN)
    feature_names = list(X_train.columns)

    class_distribution_df = (
        df[TARGET_COLUMN]
        .value_counts()
        .sort_index()
        .rename_axis("class")
        .reset_index(name="count")
    )
    class_distribution_df["percentage"] = 100 * class_distribution_df["count"] / len(df)
    class_distribution_df.to_csv(TABLES_DIR / "class_distribution.csv", index=False)

    baseline_metrics = train_baseline(X_train, y_train, X_test, y_test, feature_names)
    improved_name, improved_metrics = train_improved(
        X_train,
        y_train,
        X_test,
        y_test,
        feature_names,
        model_choice=args.improved_model,
    )

    comparison_df = pd.DataFrame(
        [
            {"model": "baseline_logistic_regression", **baseline_metrics},
            {"model": f"improved_{improved_name}", **improved_metrics},
        ]
    ).sort_values("average_precision", ascending=False)
    comparison_df.to_csv(TABLES_DIR / "model_comparison.csv", index=False)

    save_markdown_summary(
        comparison_df=comparison_df,
        class_distribution_df=class_distribution_df,
        output_path=REPORTS_DIR / "results_summary.md",
    )

    print("Training complete.")
    print(f"Model comparison table: {TABLES_DIR / 'model_comparison.csv'}")
    print(f"Report-ready summary: {REPORTS_DIR / 'results_summary.md'}")
    print(f"Models saved in: {MODELS_DIR}")


if __name__ == "__main__":
    main()
