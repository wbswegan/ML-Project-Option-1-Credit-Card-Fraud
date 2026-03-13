from typing import Optional, Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from fraud_detection.config import RANDOM_STATE

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except ImportError:
    XGBClassifier = None
    HAS_XGBOOST = False


def build_baseline_model() -> LogisticRegression:
    """Simple sklearn baseline model."""
    return LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear",
        random_state=RANDOM_STATE,
    )


def build_improved_model(
    model_choice: str = "auto",
    scale_pos_weight: Optional[float] = None,
) -> Tuple[str, object]:
    """Improved model: XGBoost when available, otherwise RandomForest."""
    choice = model_choice.lower()
    if choice not in {"auto", "xgboost", "random_forest"}:
        raise ValueError("model_choice must be one of: auto, xgboost, random_forest")

    if choice in {"auto", "xgboost"} and HAS_XGBOOST:
        if scale_pos_weight is None:
            scale_pos_weight = 1.0
        model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=1,
            scale_pos_weight=scale_pos_weight,
        )
        return "xgboost", model

    if choice == "xgboost" and not HAS_XGBOOST:
        raise ImportError("xgboost is not installed. Install it or use model_choice='random_forest'.")

    model = RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    return "random_forest", model

