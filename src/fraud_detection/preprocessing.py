from typing import List

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_preprocessor(feature_names: List[str], scale_numeric: bool = True) -> ColumnTransformer:
    """Build a preprocessing pipeline for numeric features."""
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)
    return ColumnTransformer(
        transformers=[("numeric", numeric_pipeline, feature_names)],
        remainder="drop",
    )

