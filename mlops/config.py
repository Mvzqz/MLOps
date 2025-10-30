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
# Training configuration defaults
# ---------------------------------------------------------
RANDOM_SEED = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.25
TARGET_COL = "Rented Bike Count"

# Model training defaults
DEFAULT_MODEL_NAME = "random_forest_regressor"
DEFAULT_PARAM_GRID = {
    "n_estimators": [50, 100],
    "max_depth": [5, 10],
}
DEFAULT_METRIC = "r2"
DEFAULT_SEARCH_MODE = "grid"
DEFAULT_SEARCH_PARAMS = {}
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
