from pathlib import Path

import pandas as pd


def save_markdown_summary(
    comparison_df: pd.DataFrame,
    class_distribution_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create a report-ready markdown summary of key results."""
    best_row = comparison_df.sort_values("average_precision", ascending=False).iloc[0]

    lines = [
        "# Fraud Detection Results Summary",
        "",
        "## Class Imbalance",
        class_distribution_df.to_markdown(index=False),
        "",
        "## Model Comparison",
        comparison_df.to_markdown(index=False),
        "",
        "## Key Takeaways",
        f"- Best model by Average Precision: **{best_row['model']}**",
        f"- Best model Average Precision: **{best_row['average_precision']:.4f}**",
        f"- Best model ROC-AUC: **{best_row['roc_auc']:.4f}**",
        "",
        "Use this section directly in your written report and presentation slides.",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")

