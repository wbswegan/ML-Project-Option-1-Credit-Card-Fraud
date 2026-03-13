"""Train baseline and improved fraud detection models on the training split."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from config import (
    BASELINE_MODEL_PATH,
    IMPROVED_MODEL_PATH,
    MODELS_DIR,
    PREPROCESSED_DATA_PATH,
    RANDOM_STATE,
    RAW_DATA_PATH,
    TABLES_DIR,
)
from data_preprocessing import run_preprocessing
from utils import ensure_directories, set_seed


def load_preprocessed_data(data_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load preprocessed train arrays, creating them if missing."""
    if not PREPROCESSED_DATA_PATH.exists():
        run_preprocessing(data_path)

    arrays = np.load(PREPROCESSED_DATA_PATH)
    X_train = arrays["X_train"]
    y_train = arrays["y_train"]
    return X_train, y_train


def build_baseline_model() -> LogisticRegression:
    """Create an interpretable baseline model."""
    return LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear",
        random_state=RANDOM_STATE,
    )


def build_improved_model() -> RandomForestClassifier:
    """Create a stronger tree-based model for tabular data."""
    return RandomForestClassifier(
        n_estimators=120,
        max_depth=18,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        max_samples=0.2,
    )


def train_baseline_model(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """Fit Logistic Regression on the training split."""
    model = build_baseline_model()
    model.fit(X_train, y_train)
    return model


def train_improved_model(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """Fit Random Forest on the same training split as baseline."""
    model = build_improved_model()
    model.fit(X_train, y_train)
    return model


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train baseline and improved fraud models.")
    parser.add_argument(
        "--data-path",
        default=str(RAW_DATA_PATH),
        help="Path to data/raw directory, or to a CSV file.",
    )
    return parser.parse_args()


def save_training_summary(y_train: np.ndarray) -> None:
    """Save a short training summary table for reports."""
    summary = pd.DataFrame(
        [
            {
                "model": "baseline_logistic_regression",
                "artifact": str(BASELINE_MODEL_PATH),
                "n_training_rows": int(len(y_train)),
                "training_fraud_rate": float(np.mean(y_train)),
            },
            {
                "model": "improved_random_forest",
                "artifact": str(IMPROVED_MODEL_PATH),
                "n_training_rows": int(len(y_train)),
                "training_fraud_rate": float(np.mean(y_train)),
            },
        ]
    )
    summary.to_csv(TABLES_DIR / "training_summary.csv", index=False)


def main() -> None:
    """CLI entry point for model training."""
    args = parse_args()
    set_seed(RANDOM_STATE)
    ensure_directories(MODELS_DIR, TABLES_DIR)

    X_train, y_train = load_preprocessed_data(Path(args.data_path))
    baseline_model = train_baseline_model(X_train, y_train)
    improved_model = train_improved_model(X_train, y_train)

    joblib.dump(baseline_model, BASELINE_MODEL_PATH)
    joblib.dump(improved_model, IMPROVED_MODEL_PATH)
    save_training_summary(y_train)

    print(f"Saved baseline model: {BASELINE_MODEL_PATH}")
    print(f"Saved improved model: {IMPROVED_MODEL_PATH}")
    print(f"Saved training summary: {TABLES_DIR / 'training_summary.csv'}")


if __name__ == "__main__":
    main()
