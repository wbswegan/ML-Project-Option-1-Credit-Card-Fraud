"""Run a focused EDA workflow for fraud detection datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import FIGURES_DIR, RANDOM_STATE, RAW_DATA_PATH, TABLES_DIR, TARGET_CANDIDATES
from data_preprocessing import load_data
from utils import ensure_directories, set_seed


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run EDA for the fraud dataset.")
    parser.add_argument(
        "--data-path",
        default=str(RAW_DATA_PATH),
        help="Path to data/raw directory, or to a CSV file.",
    )
    return parser.parse_args()


def infer_target_column(df: pd.DataFrame) -> str:
    """Infer target column from known fraud-label names."""
    for candidate in TARGET_CANDIDATES:
        if candidate in df.columns:
            return candidate
    raise ValueError(f"Could not infer target column from candidates: {TARGET_CANDIDATES}")


def get_amount_column(df: pd.DataFrame) -> str | None:
    """Resolve amount column name across dataset variants."""
    for candidate in ("Amount", "amt"):
        if candidate in df.columns:
            return candidate
    return None


def build_time_hour_series(df: pd.DataFrame) -> pd.Series | None:
    """Create a comparable hour-of-day series from available time columns."""
    if "trans_date_trans_time" in df.columns:
        dt = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")
        return dt.dt.hour
    if "unix_time" in df.columns:
        dt = pd.to_datetime(df["unix_time"], unit="s", errors="coerce")
        return dt.dt.hour
    if "Time" in df.columns:
        return (df["Time"] / 3600.0) % 24
    return None


def get_plot_sample(df: pd.DataFrame, max_rows: int = 200000) -> pd.DataFrame:
    """Sample large dataframes for faster plotting without changing summary tables."""
    if len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=RANDOM_STATE)


def save_inspection_tables(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Save core inspection tables for report use."""
    overview = pd.DataFrame(
        {
            "n_rows": [df.shape[0]],
            "n_columns": [df.shape[1]],
            "duplicates": [int(df.duplicated().sum())],
            "total_missing_cells": [int(df.isna().sum().sum())],
        }
    )
    overview.to_csv(TABLES_DIR / "eda_overview.csv", index=False)

    column_summary = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": df.dtypes.astype(str).values,
            "missing_count": df.isna().sum().values,
            "missing_pct": (df.isna().mean().values * 100).round(4),
            "n_unique": df.nunique(dropna=False).values,
        }
    )
    column_summary.to_csv(TABLES_DIR / "eda_column_summary.csv", index=False)

    class_distribution = (
        df[target_column]
        .value_counts()
        .sort_index()
        .rename_axis("class")
        .reset_index(name="count")
    )
    class_distribution["percentage"] = (100 * class_distribution["count"] / len(df)).round(4)
    class_distribution.to_csv(TABLES_DIR / "eda_class_distribution.csv", index=False)
    return class_distribution


def plot_class_distribution(class_distribution: pd.DataFrame) -> None:
    """Plot class imbalance with percentages."""
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=class_distribution, x="class", y="count", palette="Set2", ax=ax)
    ax.set_title("Class Distribution (0 = Non-Fraud, 1 = Fraud)")
    ax.set_xlabel("Class")
    ax.set_ylabel("Transaction Count")
    for i, row in class_distribution.iterrows():
        ax.text(i, row["count"], f"{row['percentage']:.2f}%", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_class_distribution.png", dpi=200)
    plt.close(fig)


def plot_amount_distribution(df: pd.DataFrame, target_column: str) -> dict:
    """Plot amount distributions overall and by class."""
    insights = {}
    amount_column = get_amount_column(df)
    if amount_column is None:
        return insights

    amount_df = get_plot_sample(df[[target_column, amount_column]].copy())
    amount_df["log_amount"] = np.log1p(amount_df[amount_column].clip(lower=0))

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(amount_df["log_amount"], bins=80, kde=False, color="#4c72b0", ax=ax)
    ax.set_title("Transaction Amount Distribution (log1p)")
    ax.set_xlabel("log1p(Amount)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_amount_distribution.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=amount_df, x=target_column, y="log_amount", palette="Set2", ax=ax)
    ax.set_title("Transaction Amount by Class (log1p)")
    ax.set_xlabel("Class")
    ax.set_ylabel("log1p(Amount)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_amount_by_class.png", dpi=200)
    plt.close(fig)

    grouped = df.groupby(target_column)[amount_column].median().to_dict()
    insights["median_amount_non_fraud"] = float(grouped.get(0, np.nan))
    insights["median_amount_fraud"] = float(grouped.get(1, np.nan))
    return insights


def plot_time_distribution(df: pd.DataFrame, target_column: str) -> dict:
    """Plot time-based transaction patterns if a time column exists."""
    insights = {}
    hour_series = build_time_hour_series(df)
    if hour_series is None:
        return insights

    time_df = pd.DataFrame({target_column: df[target_column], "hour": hour_series})
    time_df = time_df.dropna()
    time_df = get_plot_sample(time_df)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(
        data=time_df,
        x="hour",
        hue=target_column,
        bins=24,
        stat="density",
        common_norm=False,
        element="step",
        ax=ax,
    )
    ax.set_title("Transaction Time Distribution by Class")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Density")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_time_distribution.png", dpi=200)
    plt.close(fig)

    fraud_time = time_df[time_df[target_column] == 1]
    if not fraud_time.empty:
        fraud_peak_hour = float(fraud_time["hour"].round().value_counts().idxmax())
        insights["fraud_peak_hour_approx"] = fraud_peak_hour
    return insights


def plot_correlation_heatmap(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Plot numeric correlation heatmap and save top class correlations."""
    numeric_df = get_plot_sample(df.select_dtypes(include=[np.number]))
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(12, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        cmap="coolwarm",
        center=0.0,
        linewidths=0.2,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Correlation Heatmap (Numeric Features)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_correlation_heatmap.png", dpi=200)
    plt.close(fig)

    if target_column in corr.columns:
        corr_target = (
            corr[target_column]
            .drop(labels=[target_column], errors="ignore")
            .sort_values(key=lambda s: s.abs(), ascending=False)
        )
        top_corr = corr_target.head(10).reset_index()
        top_corr.columns = ["feature", "correlation_with_class"]
        top_corr.to_csv(TABLES_DIR / "eda_top_correlations.csv", index=False)
        return top_corr
    return pd.DataFrame(columns=["feature", "correlation_with_class"])


def plot_feature_comparison(df: pd.DataFrame, top_corr: pd.DataFrame, target_column: str) -> None:
    """Compare top correlated numeric features between fraud and non-fraud classes."""
    candidate_features = [f for f in top_corr["feature"].tolist() if f in df.columns][:4]
    if not candidate_features:
        return

    plot_df = get_plot_sample(df[[target_column] + candidate_features].copy())
    melted = plot_df.melt(id_vars=[target_column], value_vars=candidate_features)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(
        data=melted,
        x="variable",
        y="value",
        hue=target_column,
        showfliers=False,
        ax=ax,
    )
    ax.set_title("Top Fraud-Relevant Features: Fraud vs Non-Fraud")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Value")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="Class")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_feature_comparison.png", dpi=200)
    plt.close(fig)


def save_findings_markdown(
    df: pd.DataFrame,
    class_distribution: pd.DataFrame,
    top_corr: pd.DataFrame,
    amount_insights: dict,
    time_insights: dict,
) -> None:
    """Save brief, report-friendly findings as markdown."""
    fraud_rate = float(
        class_distribution.loc[class_distribution["class"] == 1, "percentage"].iloc[0]
        if (class_distribution["class"] == 1).any()
        else np.nan
    )
    non_fraud_rate = float(
        class_distribution.loc[class_distribution["class"] == 0, "percentage"].iloc[0]
        if (class_distribution["class"] == 0).any()
        else np.nan
    )

    top_features = ", ".join(top_corr["feature"].head(4).tolist()) if not top_corr.empty else "N/A"
    median_non_fraud = amount_insights.get("median_amount_non_fraud", np.nan)
    median_fraud = amount_insights.get("median_amount_fraud", np.nan)
    fraud_peak_hour = time_insights.get("fraud_peak_hour_approx", None)

    findings = [
        "# EDA Findings (Fraud Detection)",
        "",
        "1. **Strong class imbalance**",
        f"- Non-fraud: {non_fraud_rate:.4f}% | Fraud: {fraud_rate:.4f}% of all transactions.",
        "",
        "2. **Data quality checks**",
        f"- Duplicate rows: {int(df.duplicated().sum())}",
        f"- Total missing cells: {int(df.isna().sum().sum())}",
        "",
        "3. **Amount behavior**",
        (
            f"- Median amount (non-fraud): {median_non_fraud:.2f} | "
            f"Median amount (fraud): {median_fraud:.2f}"
        ),
        "",
        "4. **Most class-related numeric features**",
        f"- Top correlated features with fraud label: {top_features}",
    ]

    if fraud_peak_hour is not None:
        findings.extend(
            [
                "",
                "5. **Time pattern**",
                f"- Approximate fraud activity peak at hour: {fraud_peak_hour:.0f}",
            ]
        )

    findings.extend(
        [
            "",
            "These findings support using imbalance-aware metrics such as recall and F1 in modeling.",
        ]
    )

    (TABLES_DIR / "eda_findings.md").write_text("\n".join(findings), encoding="utf-8")


def main() -> None:
    """Execute EDA workflow and save outputs."""
    args = parse_args()
    set_seed(RANDOM_STATE)
    ensure_directories(FIGURES_DIR, TABLES_DIR)

    sns.set_theme(style="whitegrid")
    df = load_data(Path(args.data_path))
    target_column = infer_target_column(df)

    class_distribution = save_inspection_tables(df, target_column=target_column)
    plot_class_distribution(class_distribution)
    amount_insights = plot_amount_distribution(df, target_column=target_column)
    time_insights = plot_time_distribution(df, target_column=target_column)
    top_corr = plot_correlation_heatmap(df, target_column=target_column)
    plot_feature_comparison(df, top_corr, target_column=target_column)
    save_findings_markdown(df, class_distribution, top_corr, amount_insights, time_insights)

    print("EDA workflow complete.")
    print(f"Saved figures: {FIGURES_DIR}")
    print(f"Saved tables/findings: {TABLES_DIR}")


if __name__ == "__main__":
    main()

