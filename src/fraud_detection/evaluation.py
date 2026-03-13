from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(y_true, y_pred, y_prob) -> Dict[str, float]:
    """Compute key binary classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "average_precision": average_precision_score(y_true, y_prob),
    }


def save_classification_report(y_true, y_pred, output_path: Path) -> None:
    """Save sklearn classification report as CSV."""
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).T
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(output_path)


def plot_confusion(y_true, y_pred, output_path: Path, model_name: str) -> None:
    """Save confusion matrix figure."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax, colorbar=False)
    ax.set_title(f"{model_name} - Confusion Matrix")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_roc(y_true, y_prob, output_path: Path, model_name: str) -> None:
    """Save ROC curve figure."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{model_name} - ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_precision_recall(y_true, y_prob, output_path: Path, model_name: str) -> None:
    """Save precision-recall curve figure."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap_score = average_precision_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"AP = {ap_score:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{model_name} - Precision Recall Curve")
    ax.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_probability_histogram(y_true, y_prob, output_path: Path, model_name: str) -> None:
    """Plot predicted probability distributions by class."""
    df_plot = pd.DataFrame({"label": y_true, "probability": y_prob})
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(
        data=df_plot,
        x="probability",
        hue="label",
        bins=50,
        stat="density",
        common_norm=False,
        ax=ax,
    )
    ax.set_title(f"{model_name} - Predicted Probability Distribution")
    ax.set_xlabel("Predicted fraud probability")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

