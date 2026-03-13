from pathlib import Path
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from fraud_detection.config import (  # noqa: E402
    FIGURES_DIR,
    RANDOM_STATE,
    RAW_DATA_PATH,
    TABLES_DIR,
    TARGET_COLUMN,
)
from fraud_detection.data import load_dataset  # noqa: E402
from fraud_detection.utils import ensure_directories, set_global_seed  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Run EDA for credit card fraud detection dataset.")
    parser.add_argument(
        "--data-path",
        default=str(RAW_DATA_PATH),
        help="Path to creditcard.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(RANDOM_STATE)
    ensure_directories([FIGURES_DIR, TABLES_DIR])

    df = load_dataset(Path(args.data_path))

    # 1) Dataset overview table
    overview = pd.DataFrame(
        {
            "n_rows": [df.shape[0]],
            "n_columns": [df.shape[1]],
            "n_duplicates": [int(df.duplicated().sum())],
            "n_missing_cells": [int(df.isna().sum().sum())],
        }
    )
    overview.to_csv(TABLES_DIR / "eda_overview.csv", index=False)

    # 2) Missing values table
    missing_df = (
        df.isna()
        .sum()
        .rename("missing_count")
        .reset_index()
        .rename(columns={"index": "feature"})
        .sort_values("missing_count", ascending=False)
    )
    missing_df["missing_pct"] = 100 * missing_df["missing_count"] / len(df)
    missing_df.to_csv(TABLES_DIR / "missing_values.csv", index=False)

    # 3) Class imbalance table
    class_distribution = (
        df[TARGET_COLUMN]
        .value_counts()
        .sort_index()
        .rename_axis("class")
        .reset_index(name="count")
    )
    class_distribution["percentage"] = 100 * class_distribution["count"] / len(df)
    class_distribution.to_csv(TABLES_DIR / "class_distribution.csv", index=False)

    # 4) Numeric summary table
    numeric_summary = df.describe().T.reset_index().rename(columns={"index": "feature"})
    numeric_summary.to_csv(TABLES_DIR / "numeric_summary.csv", index=False)

    # Figure: class distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=class_distribution, x="class", y="count", palette="Set2", ax=ax)
    ax.set_title("Class Distribution (0=Normal, 1=Fraud)")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_class_distribution.png", dpi=200)
    plt.close(fig)

    # Figure: log-transformed amount by class
    amount_plot = df[[TARGET_COLUMN, "Amount"]].copy()
    amount_plot["log_amount"] = np.log1p(amount_plot["Amount"])
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=amount_plot, x=TARGET_COLUMN, y="log_amount", palette="Set2", ax=ax)
    ax.set_title("Log(Amount) by Class")
    ax.set_xlabel("Class")
    ax.set_ylabel("log1p(Amount)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_amount_by_class.png", dpi=200)
    plt.close(fig)

    # Figure: transaction time histogram by class
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(
        data=df,
        x="Time",
        hue=TARGET_COLUMN,
        bins=80,
        element="step",
        stat="density",
        common_norm=False,
        ax=ax,
    )
    ax.set_title("Transaction Time Distribution by Class")
    ax.set_xlabel("Time")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_time_distribution.png", dpi=200)
    plt.close(fig)

    # Figure + table: top correlations with target
    corr_series = (
        df.corr(numeric_only=True)[TARGET_COLUMN]
        .drop(labels=[TARGET_COLUMN], errors="ignore")
        .sort_values(key=lambda s: s.abs(), ascending=False)
        .head(12)
    )
    corr_df = corr_series.reset_index()
    corr_df.columns = ["feature", "correlation_with_class"]
    corr_df.to_csv(TABLES_DIR / "top_feature_correlations.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=corr_df,
        x="correlation_with_class",
        y="feature",
        palette="viridis",
        ax=ax,
    )
    ax.set_title("Top Correlations with Fraud Label")
    ax.set_xlabel("Correlation with Class")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_top_correlations.png", dpi=200)
    plt.close(fig)

    print("EDA complete.")
    print(f"Saved tables to: {TABLES_DIR}")
    print(f"Saved figures to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
