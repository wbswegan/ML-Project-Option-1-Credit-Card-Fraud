from pathlib import Path
import argparse
import sys

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from fraud_detection.config import (  # noqa: E402
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    RAW_DATA_PATH,
    TARGET_COLUMN,
)
from fraud_detection.data import load_dataset, split_data  # noqa: E402
from fraud_detection.preprocessing import build_preprocessor  # noqa: E402
from fraud_detection.utils import ensure_directories, set_global_seed  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Fit and apply preprocessing pipeline.")
    parser.add_argument(
        "--data-path",
        default=str(RAW_DATA_PATH),
        help="Path to creditcard.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(RANDOM_STATE)
    ensure_directories([PROCESSED_DATA_DIR, MODELS_DIR])

    df = load_dataset(Path(args.data_path))
    X_train, X_test, y_train, y_test = split_data(df, target_column=TARGET_COLUMN)
    feature_names = list(X_train.columns)

    preprocessor = build_preprocessor(feature_names=feature_names, scale_numeric=True)
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    np.savez_compressed(
        PROCESSED_DATA_DIR / "preprocessed_data.npz",
        X_train=X_train_transformed,
        X_test=X_test_transformed,
        y_train=y_train.to_numpy(),
        y_test=y_test.to_numpy(),
    )

    split_info = pd.DataFrame(
        [
            {"split": "train", "n_rows": len(X_train), "fraud_rate": y_train.mean()},
            {"split": "test", "n_rows": len(X_test), "fraud_rate": y_test.mean()},
        ]
    )
    split_info.to_csv(PROCESSED_DATA_DIR / "split_info.csv", index=False)

    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")

    print("Preprocessing complete.")
    print(f"Saved processed arrays to: {PROCESSED_DATA_DIR / 'preprocessed_data.npz'}")
    print(f"Saved split summary to: {PROCESSED_DATA_DIR / 'split_info.csv'}")
    print(f"Saved fitted preprocessor to: {MODELS_DIR / 'preprocessor.joblib'}")


if __name__ == "__main__":
    main()
