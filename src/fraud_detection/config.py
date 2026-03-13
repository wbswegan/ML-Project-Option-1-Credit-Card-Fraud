from pathlib import Path

# Reproducibility
RANDOM_STATE = 42

# Dataset settings
TARGET_COLUMN = "Class"
TEST_SIZE = 0.20

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
MODELS_DIR = OUTPUT_DIR / "models"
METRICS_DIR = OUTPUT_DIR / "metrics"
REPORTS_DIR = OUTPUT_DIR / "reports"

