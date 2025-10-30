from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# ---------------------------------------------------------
# Core training configuration
# ---------------------------------------------------------
RANDOM_SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15
TARGET_COL = "Rented Bike Count"

# ---------------------------------------------------------
# Default model and search configuration
# ---------------------------------------------------------
# For Seoul Bike Sharing or similar regression datasets
DEFAULT_MODEL_NAME = "hist_gradient_boosting_regressor"

# Smaller but meaningful grid to ensure success and prevent overfitting/timeouts
DEFAULT_PARAM_GRID = {
    "max_depth": [5, 10, None],
    "learning_rate": [0.05, 0.1],
    "max_iter": [100, 200],
    "min_samples_leaf": [10, 20],
    "l2_regularization": [0.0, 0.1],
}

DEFAULT_METRIC = "r2"  # r2 works well for regression
DEFAULT_SEARCH_MODE = "halving_grid"  # faster & more efficient than full grid
DEFAULT_SEARCH_PARAMS = {
    "factor": 3,
    "min_resources": "exhaust",
    "random_state": RANDOM_SEED,
}
DEFAULT_CV = 3
DEFAULT_MODEL_PATH = MODELS_DIR / "best_model.pkl"

# ---------------------------------------------------------
# Fallbacks for classification tasks (optional)
# ---------------------------------------------------------
CLASSIFICATION_FALLBACKS = {
    "model": "random_forest_classifier",
    "param_grid": {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5],
    },
    "metric": "f1",
}

# ---------------------------------------------------------
# tqdm-safe logger configuration
# ---------------------------------------------------------
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
