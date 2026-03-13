from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from fraud_detection.config import RANDOM_STATE, TARGET_COLUMN, TEST_SIZE


def load_dataset(data_path: Path) -> pd.DataFrame:
    """Load the credit card fraud dataset from CSV."""
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{data_path}'. "
            "Download `creditcard.csv` from Kaggle and place it in data/raw/."
        )

    df = pd.read_csv(data_path)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' was not found in the dataset.")
    return df


def split_data(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    test_size: float = TEST_SIZE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split dataframe into train and test sets using stratification."""
    X = df.drop(columns=[target_column])
    y = df[target_column].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test

