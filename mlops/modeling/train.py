"""Training module for the MLOps project with MLflow and DagsHub integration."""

import json
import pickle
import os
from pathlib import Path
from typing import Any, Dict, Optional

import dagshub
from dotenv import load_dotenv
from loguru import logger
import mlflow
from mlflow.data.dataset_source import DatasetSource
from mlflow.data.pandas_dataset import PandasDataset
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    GridSearchCV,
    HalvingGridSearchCV,
    HalvingRandomSearchCV,
    TimeSeriesSplit,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVR
import typer

from mlops.config import (
    DEFAULT_CV,
    DEFAULT_METRIC,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_PATH,
    DEFAULT_MODEL_PATH, PARAM_GRIDS,
    DEFAULT_PARAM_GRID,
    DEFAULT_SEARCH_MODE,
    DEFAULT_SEARCH_PARAMS,
    PROCESSED_DATA_DIR,
    TARGET_COL,
    setup_mlflow_connection,
)

app = typer.Typer()

# -------------------------------------------------------------------
# Initialize MLflow connection
# -------------------------------------------------------------------
setup_mlflow_connection()

# -------------------------------------------------------------------
# MODEL AND SEARCH REGISTRIES
# -------------------------------------------------------------------
MODEL_REGISTRY = {
    "hist_gradient_boosting_regressor": HistGradientBoostingRegressor,
    "random_forest_regressor": RandomForestRegressor,
    "xgb_regressor": XGBRegressor,
}

SEARCH_REGISTRY = {
    "grid": GridSearchCV,
    "halving_grid": HalvingGridSearchCV,
    "halving_random": HalvingRandomSearchCV,
}


class ModelTrainer:
    """Encapsulates the model training, tuning, and evaluation pipeline."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None

    def _load_and_split_data(self):
        """Loads and splits the data chronologically."""
        path = self.config["dataset_path"]
        logger.info(f"Loading dataset from {path}...")
        df = pd.read_csv(path).sort_values("year").reset_index(drop=True)

        target_col = self.config["target_col"]
        if target_col not in df.columns:
            raise ValueError(
                f"Target column '{target_col}' not found. Columns: {df.columns.tolist()}"
            )
        # Drop rows where the target column is NaN to avoid errors during training
        initial_rows = len(df)
        df.dropna(subset=[target_col], inplace=True)
        if len(df) < initial_rows:
            logger.warning(
                f"Dropped {initial_rows - len(df)} rows with null values in the target column '{target_col}'."
            )

        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Time-based split to prevent data leakage
        split_index = int(len(df) * (1 - self.config["test_size"]))
        X_train, self.X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, self.y_test = y.iloc[:split_index], y.iloc[split_index:]

        # Apply log1p transformation to the target variable
        logger.info("Applying log1p transformation to the target variable.")
        self.y_train = np.log1p(y_train)
        # We keep y_test in its original scale for evaluation

        # Assign training features
        self.X_train = X_train

        logger.info(f"Train: {len(X_train)} / Test: {len(self.X_test)} rows.")
        return self

    def _build_pipeline(self):
        """Builds the preprocessing and model pipeline."""
        model_name = self.config["model_name"]
        if model_name not in MODEL_REGISTRY:
            logger.error(
                f"Model '{model_name}' not found. Available: {list(MODEL_REGISTRY.keys())}"
            )
            raise ValueError(f"Model '{model_name}' not available.")

        model_instance = MODEL_REGISTRY[model_name]()
        cat_cols = self.X_train.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = self.X_train.select_dtypes(include=np.number).columns.tolist()
        logger.info(f"Categorical columns: {cat_cols}")
        logger.info(f"Numerical columns: {num_cols}")

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", SimpleImputer(strategy="median"), num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ]
        )
        return Pipeline(steps=[("preprocessor", preprocessor), ("model", model_instance)])

    def _build_search_strategy(self, pipeline: Pipeline):
        """Configures the hyperparameter search."""
        search_mode = self.config["search_mode"]
        search_class = SEARCH_REGISTRY[search_mode]

        # Filter search parameters to pass only the relevant ones
        # for the selected search class.
        valid_search_params = {
            "grid": [],
            "halving_grid": ["factor", "min_resources", "random_state"],
            "halving_random": ["factor", "min_resources", "random_state"],
        }

        relevant_params = {
            k: v for k, v in self.config["search_params"].items()
            if k in valid_search_params.get(search_mode, [])
        }

        time_series_cv = TimeSeriesSplit(n_splits=self.config["cv"])
        return search_class(
            estimator=pipeline,
            param_grid=self.config["param_grid"],
            scoring=self.config["metric"],
            cv=time_series_cv,
            n_jobs=-1,
            verbose=1,
            **relevant_params,
        )

    def run(self):
        """Runs the complete training and MLflow logging pipeline."""
        self._load_and_split_data()

        # --- Create MLflow run ---
        with mlflow.start_run(run_name=self.config["run_name"], experiment_id=self.config["experiment_id"]):
            mlflow.log_params(
                {k: v for k, v in self.config.items() if isinstance(v, (int, float, str))}
            )

            pipeline = self._build_pipeline()
            search = self._build_search_strategy(pipeline)
            logger.info("Starting hyperparameter search...")
            search.fit(self.X_train, self.y_train)
            logger.info("Hyperparameter search completed.")
            # Log dataset schema
            schema = {
                "num_features": len(self.X_train.columns),
                "columns": self.X_train.columns.tolist(),
                "dtypes": self.X_train.dtypes.astype(str).to_dict(),
                "target_col": self.config["target_col"],
            }
            schema_path = Path("dataset_schema.json")
            with open(schema_path, "w") as f:
                json.dump(schema, f, indent=2)
            mlflow.log_artifact(str(schema_path), artifact_path="metadata")

            # --- Log Dataset to MLflow ---
            # Create a full DataFrame for logging, ensuring the target column is a Series
            full_df = pd.concat([self.X_train, self.X_test], axis=0).sort_index()
            full_df[self.config["target_col"]] = pd.concat([self.y_train, self.y_test], axis=0).sort_index()
            dataset = mlflow.data.from_pandas(
                full_df,
                source=str(self.config["dataset_path"]),
                name="seoul-bike-sharing-featured",
                targets=self.config["target_col"]
            )
            # Log the dataset as an input for this run
            mlflow.log_input(dataset, context="training")

            best_model = search.best_estimator_
            best_params = search.best_params_
            best_cv_score = search.best_score_

            # Log CV metrics
            mlflow.log_metrics({"cv_best_score": best_cv_score})
            mlflow.log_params(best_params)

            # Evaluate on test set
            y_pred_log = best_model.predict(self.X_test)
            # Inverse transform predictions to original scale for evaluation
            y_pred = np.expm1(y_pred_log)

            test_r2 = r2_score(self.y_test, y_pred)
            test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))

            mlflow.log_metrics({"test_r2": test_r2, "test_rmse": test_rmse})
            logger.info(f"Test results - R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}")

            # The model is now saved only in MLflow, not locally at this stage.
            mlflow.sklearn.log_model(best_model, artifact_path="model")

            logger.success(
                f"Model '{self.config['model_name']}' saved and logged to MLflow."
            )


# -------------------------------------------------------------------
# Main training pipeline
# -------------------------------------------------------------------
@app.command()
def main(
    run_name: str = typer.Option("default_run", help="Name of the MLflow run."),
    dataset_path: Path = PROCESSED_DATA_DIR / "seoul_bike_sharing_featured.csv",
    target_col: str = TARGET_COL,
    model_name: str = DEFAULT_MODEL_NAME,
    param_grid: Optional[str] = None,
    metric: str = DEFAULT_METRIC,
    search_mode: str = DEFAULT_SEARCH_MODE,
    search_params: str = json.dumps(DEFAULT_SEARCH_PARAMS),
    cv: int = DEFAULT_CV,
    test_size: float = 0.2,
    model_path: Path = DEFAULT_MODEL_PATH,
):
    """Trains, tunes, and evaluates a model using MLflow + DagsHub."""
    # Ensure the experiment exists in MLflow
    experiment_name = "bike_demand_prediction"
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    # If no parameter grid is provided, use the default for the model.
    if param_grid is None:
        grid = PARAM_GRIDS.get(model_name, DEFAULT_PARAM_GRID)
    else:
        grid = json.loads(param_grid)
    training_config = {
        "run_name": run_name,
        "dataset_path": dataset_path,
        "experiment_id": experiment.experiment_id,
        "target_col": target_col,
        "model_name": model_name,
        "param_grid": grid,
        "metric": metric,
        "search_mode": search_mode,
        "search_params": json.loads(search_params),
        "cv": cv,
        "test_size": test_size,
        "model_path": model_path,
    }
    trainer = ModelTrainer(config=training_config)
    trainer.run()


if __name__ == "__main__":
    app()
