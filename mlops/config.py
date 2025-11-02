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
TEST_SIZE = 0.10
VAL_SIZE = 0.10
TARGET_COL = "rented_bike_count"

# ---------------------------------------------------------
# Default model and search configuration
# ---------------------------------------------------------
# For Seoul Bike Sharing or similar regression datasets
DEFAULT_MODEL_NAME = "hist_gradient_boosting_regressor"

# Default parameter grids for different models
PARAM_GRIDS = {
    "hist_gradient_boosting_regressor": {
        "model__max_depth": [10, 12],
        "model__learning_rate": [0.08, 0.1],
        "model__max_iter": [400, 500],
        "model__l2_regularization": [0.0, 0.5],
    },
    "random_forest_regressor": {
        "model__n_estimators": [500],
        "model__random_state": [42],
        "model__min_samples_leaf": [2],
        "model__max_features": ["sqrt"],
        "model__n_jobs": [-1],
        "model__max_depth": [20],
    },
    "xgb_regressor": {
        "model__n_estimators": [700],
        "model__max_depth": [6],
        "model__learning_rate": [0.03],
        "model__random_state": [42],
        "model__subsample": [0.8],
        "model__colsample_bytree": [0.8],
        "model__reg_lambda": [1.0],
        "model__n_jobs": [-1],
        "model__tree_method": ["hist"],
    },
}

# Default param_grid if none is specified for a model
DEFAULT_PARAM_GRID = PARAM_GRIDS.get(DEFAULT_MODEL_NAME, {})

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
# tqdm-safe logger configuration
# ---------------------------------------------------------
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
