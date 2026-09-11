"""Central configuration for the Used Car Price Prediction project.

Every path, column list and random seed used anywhere in the project is
defined here so that Weeks 5-15 stay consistent with one another.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"
PLOT_DIR = REPORT_DIR / "plots"
MODEL_DIR = ROOT / "models"
DOCS_DIR = ROOT / "docs"

RAW_PATH = DATA_DIR / "car_data_raw.csv"
CLEAN_PATH = DATA_DIR / "car_data_cleaned.csv"

BUNDLE_PATH = MODEL_DIR / "final_model.joblib"
BASELINE_DIR = MODEL_DIR

TARGET = "selling_price"
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# Columns of the supplied CarDekho CSV (Week 5 data dictionary).
BASE_COLUMNS = [
    "name",
    "year",
    "selling_price",
    "km_driven",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
]

RAW_NUMERIC_COLUMNS = ["year", "selling_price", "km_driven"]
RAW_TEXT_COLUMNS = ["name", "fuel", "seller_type", "transmission", "owner"]

# Features created in Week 8.
ENGINEERED_COLUMNS = [
    "car_age",
    "brand",
    "model_family",
    "km_per_year",
    "name_frequency",
]

NUMERIC_FEATURES = [
    "year",
    "km_driven",
    "car_age",
    "km_per_year",
    "name_frequency",
]

CATEGORICAL_FEATURES = [
    "name",
    "brand",
    "model_family",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Validation bounds used by Week 6 invalid-value checking.
MIN_VALID_YEAR = 1950
MAX_VALID_YEAR = 2030

# Documented project objective (Week 1/2). The code measures and reports the
# real score against this number; it never asserts or fabricates it.
TARGET_R2 = 0.90


def ensure_directories():
    """Create every output directory the pipeline writes to."""
    for directory in (DATA_DIR, REPORT_DIR, PLOT_DIR, MODEL_DIR, DOCS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
