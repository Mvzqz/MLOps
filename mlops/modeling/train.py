"""Training module for the MLOps project."""

import json
from pathlib import Path
import pickle

from loguru import logger
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    HalvingGridSearchCV,
    HalvingRandomSearchCV,
    train_test_split,
)
from sklearn.svm import SVC, SVR
from tqdm import tqdm
import typer
from xgboost import XGBClassifier, XGBRegressor

from mlops.config import (
    DEFAULT_CV,
    DEFAULT_METRIC,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_PATH,
    DEFAULT_PARAM_GRID,
    DEFAULT_SEARCH_MODE,
    DEFAULT_SEARCH_PARAMS,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    TARGET_COL,
    TEST_SIZE,
    VAL_SIZE,
)

app = typer.Typer()

# -------------------------------------------------------------------
# MODEL AND SEARCH REGISTRIES
# -------------------------------------------------------------------
MODEL_REGISTRY = {
    "logistic_regression": LogisticRegression,
    "svc": SVC,
    "random_forest_classifier": RandomForestClassifier,
    "xgb_classifier": XGBClassifier,
    "linear_regression": LinearRegression,
    "svr": SVR,
    "random_forest_regressor": RandomForestRegressor,
    "xgb_regressor": XGBRegressor,
    "hist_gradient_boosting_regressor": HistGradientBoostingRegressor,
}

SEARCH_REGISTRY = {
    "grid": GridSearchCV,
    "halving_grid": HalvingGridSearchCV,
    "halving_random": HalvingRandomSearchCV,
}


# -------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------
def parse_json_arg(arg_name: str, arg_value: str):
    try:
        return json.loads(arg_value)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON for {arg_name}: {e}")
        raise typer.Exit(code=1)


def load_data(dataset_path: Path, target_col: str, val_size: float, test_size: float):
    logger.info(f"Loading dataset from {dataset_path}")
    df = pd.read_csv(dataset_path)
    if target_col not in df.columns:
        logger.error(f"Target column '{target_col}' not found. Available: {list(df.columns)}")
        raise typer.Exit(code=1)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_SEED
    )
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=RANDOM_SEED
    )

    logger.info(f"Split sizes: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_model(model_name: str):
    if model_name not in MODEL_REGISTRY:
        logger.error(f"Model '{model_name}' not found. Available: {list(MODEL_REGISTRY.keys())}")
        raise typer.Exit(code=1)
    logger.info(f"Using model: {model_name}")
    return MODEL_REGISTRY[model_name]()


def build_search_strategy(search_mode: str, model_instance, param_grid, metric, cv, search_params):
    if search_mode not in SEARCH_REGISTRY:
        logger.error(
            f"Unsupported search mode '{search_mode}'. Choose from {list(SEARCH_REGISTRY.keys())}"
        )
        raise typer.Exit(code=1)

    search_class = SEARCH_REGISTRY[search_mode]
    logger.info(f"Initializing search strategy: {search_class.__name__}")
    return search_class(
        estimator=model_instance,
        param_grid=param_grid,
        scoring=metric,
        cv=cv,
        n_jobs=-1,
        verbose=1,
        **search_params,
    )


def evaluate_model(model, X, y, metric: str):
    y_pred = model.predict(X)
    if metric in ["accuracy", "precision", "recall", "f1"]:
        return {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y, y_pred, average="weighted", zero_division=0),
        }
    elif metric in ["mse", "r2"]:
        return {"mse": mean_squared_error(y, y_pred), "r2": r2_score(y, y_pred)}
    else:
        logger.error(f"Unsupported metric: {metric}")
        raise typer.Exit(code=1)


def save_model(model, model_path: Path):
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.success(f"Best model saved at {model_path}")


# -------------------------------------------------------------------
# Main training pipeline
# -------------------------------------------------------------------
@app.command()
def main(
    dataset_path: Path = PROCESSED_DATA_DIR / "features.csv",
    target_col: str = TARGET_COL,
    model_name: str = DEFAULT_MODEL_NAME,
    param_grid: str = json.dumps(DEFAULT_PARAM_GRID),
    metric: str = DEFAULT_METRIC,
    search_mode: str = DEFAULT_SEARCH_MODE,
    search_params: str = json.dumps(DEFAULT_SEARCH_PARAMS),
    cv: int = DEFAULT_CV,
    val_size: float = VAL_SIZE,
    test_size: float = TEST_SIZE,
    model_path: Path = DEFAULT_MODEL_PATH,
):
    """Train, tune, and evaluate a model using defaults from config."""

    param_grid = parse_json_arg("param_grid", param_grid)
    search_params = parse_json_arg("search_params", search_params)

    X_train, X_val, X_test, y_train, y_val, y_test = load_data(
        dataset_path, target_col, val_size, test_size
    )

    model_instance = build_model(model_name)
    search = build_search_strategy(
        search_mode, model_instance, param_grid, metric, cv, search_params
    )

    logger.info(f"Starting hyperparameter search for {model_name}...")
    with tqdm(total=1, desc="Hyperparameter Search"):
        search.fit(X_train, y_train)

    best_model = search.best_estimator_
    best_params = search.best_params_
    logger.success(f"Best parameters: {best_params}")

    val_results = evaluate_model(best_model, X_val, y_val, metric)
    logger.info(f"Validation results: {val_results}")

    test_results = evaluate_model(best_model, X_test, y_test, metric)
    logger.info(f"Test results: {test_results}")

    save_model(best_model, model_path)

    typer.echo("===== TRAINING SUMMARY =====")
    typer.echo(f"Model: {model_name}")
    typer.echo(f"Target column: {target_col}")
    typer.echo(f"Search mode: {search_mode}")
    typer.echo(f"Best Params: {best_params}")
    typer.echo(f"Validation Metrics: {val_results}")
    typer.echo(f"Test Metrics: {test_results}")


if __name__ == "__main__":
    app()
