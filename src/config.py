"""Central configuration for paths and reproducibility."""

from pathlib import Path

RANDOM_STATE = 42
TARGET_COLUMN = "is_fraud"
TARGET_CANDIDATES = ("is_fraud", "Class")
TEST_SIZE = 0.2

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_TRAIN_PATH = DATA_RAW_DIR / "fraudTrain.csv"
RAW_TEST_PATH = DATA_RAW_DIR / "fraudTest.csv"
RAW_DATA_PATH = DATA_RAW_DIR
PREPROCESSED_DATA_PATH = DATA_PROCESSED_DIR / "preprocessed_data.npz"
PREPROCESSOR_PATH = DATA_PROCESSED_DIR / "preprocessor.joblib"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
MODELS_DIR = OUTPUT_DIR / "models"

BASELINE_MODEL_PATH = MODELS_DIR / "baseline_logistic_regression.joblib"
IMPROVED_MODEL_PATH = MODELS_DIR / "improved_random_forest.joblib"
METRICS_TABLE_PATH = TABLES_DIR / "model_metrics.csv"
MODEL_COMPARISON_TABLE_PATH = TABLES_DIR / "model_comparison.csv"
MODEL_COMPARISON_SUMMARY_PATH = TABLES_DIR / "model_comparison_summary.md"
CV_COMPARISON_TABLE_PATH = TABLES_DIR / "cv_model_comparison.csv"
VALIDATION_NOTES_PATH = TABLES_DIR / "validation_leakage_notes.md"
EVALUATION_METRICS_SUMMARY_PATH = TABLES_DIR / "evaluation_metrics_summary.csv"
EVALUATION_INTERPRETATION_PATH = TABLES_DIR / "evaluation_interpretation.md"
ROC_COMPARISON_FIGURE_PATH = FIGURES_DIR / "model_roc_comparison.png"
