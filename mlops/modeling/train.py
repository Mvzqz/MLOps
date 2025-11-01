"""Training module for the MLOps project with MLflow and DagsHub integration."""

import json
import os
from pathlib import Path
import pickle
from typing import Any, Dict, Optional

import dagshub
from dotenv import load_dotenv
from loguru import logger
import mlflow
from mlflow.data.dataset_source import DatasetSource
from mlflow.data.pandas_dataset import PandasDataset
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
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
    DEFAULT_PARAM_GRID,
    DEFAULT_SEARCH_MODE,
    DEFAULT_SEARCH_PARAMS,
    PROCESSED_DATA_DIR,
    TARGET_COL,
)

app = typer.Typer()

# -------------------------------------------------------------------
# Load environment and initialize DagsHub connection
# -------------------------------------------------------------------
load_dotenv()

dagshub_user = os.getenv("DAGSHUB_USERNAME")
dagshub_repo = os.getenv("DAGSHUB_REPO")

if dagshub_user and dagshub_repo:
    dagshub.init(repo_owner=dagshub_user, repo_name=dagshub_repo, mlflow=True)
    mlflow.set_tracking_uri(f"https://dagshub.com/{dagshub_user}/{dagshub_repo}.mlflow")
    logger.info(f"MLflow tracking set to DagsHub: {dagshub_user}/{dagshub_repo}")
else:
    logger.warning(
        "DAGSHUB_USER or DAGSHUB_REPO not found in environment. Using local MLflow tracking."
    )

# -------------------------------------------------------------------
# MODEL AND SEARCH REGISTRIES
# -------------------------------------------------------------------
MODEL_REGISTRY = {
    "logistic_regression": LogisticRegression,
    "linear_regression": LinearRegression,
    "svr": SVR,
    "hist_gradient_boosting_regressor": HistGradientBoostingRegressor,
}

SEARCH_REGISTRY = {
    "grid": GridSearchCV,
    "halving_grid": HalvingGridSearchCV,
    "halving_random": HalvingRandomSearchCV,
}


class ModelTrainer:
    """Encapsula el pipeline de entrenamiento, tuneo y evaluación del modelo."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None

    def _load_and_split_data(self):
        """Carga los datos y los divide temporalmente."""
        path = self.config["dataset_path"]
        logger.info(f"Cargando dataset desde {path}...")
        df = pd.read_csv(path).sort_values("year").reset_index(drop=True)

        target_col = self.config["target_col"]
        if target_col not in df.columns:
            raise ValueError(
                f"Columna objetivo '{target_col}' no encontrada. Columnas: {df.columns.tolist()}"
            )
        # Eliminar filas donde la columna objetivo es NaN para evitar errores en el entrenamiento
        initial_rows = len(df)
        df.dropna(subset=[target_col], inplace=True)
        if len(df) < initial_rows:
            logger.warning(
                f"Se eliminaron {initial_rows - len(df)} filas con valores nulos en la columna objetivo '{target_col}'."
            )

        X = df.drop(columns=[target_col])
        y = df[target_col]

        # División temporal para evitar fuga de datos
        split_index = int(len(df) * (1 - self.config["test_size"]))
        self.X_train, self.X_test = X.iloc[:split_index], X.iloc[split_index:]
        self.y_train, self.y_test = y.iloc[:split_index], y.iloc[split_index:]

        logger.info(f"Train: {len(self.X_train)} / Test: {len(self.X_test)} filas.")
        return self

    def _build_pipeline(self):
        """Construye el pipeline de preprocesamiento y modelo."""
        model_name = self.config["model_name"]
        if model_name not in MODEL_REGISTRY:
            logger.error(
                f"Modelo '{model_name}' no encontrado. Disponibles: {list(MODEL_REGISTRY.keys())}"
            )
            raise ValueError(f"Modelo '{model_name}' no disponible.")

        model_instance = MODEL_REGISTRY[model_name]()
        cat_cols = self.X_train.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = self.X_train.select_dtypes(include=np.number).columns.tolist()
        logger.info(f"Columnas categóricas: {cat_cols}")
        logger.info(f"Columnas numéricas: {num_cols}")

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", SimpleImputer(strategy="median"), num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ]
        )
        return Pipeline(steps=[("preprocessor", preprocessor), ("model", model_instance)])

    def _build_search_strategy(self, pipeline: Pipeline):
        """Configura la búsqueda de hiperparámetros."""
        search_mode = self.config["search_mode"]
        search_class = SEARCH_REGISTRY[search_mode]
        time_series_cv = TimeSeriesSplit(n_splits=self.config["cv"])
        return search_class(
            estimator=pipeline,
            param_grid=self.config["param_grid"],
            scoring=self.config["metric"],
            cv=time_series_cv,
            n_jobs=-1,
            verbose=1,
            **self.config["search_params"],
        )

    def run(self):
        """Ejecuta el pipeline completo de entrenamiento y logging en MLflow."""
        self._load_and_split_data()

        # --- Create MLflow run ---
        with mlflow.start_run(run_name=self.config["model_name"]):
            mlflow.log_params(
                {k: v for k, v in self.config.items() if isinstance(v, (int, float, str))}
            )

            pipeline = self._build_pipeline()
            search = self._build_search_strategy(pipeline)
            logger.info("Iniciando búsqueda de hiperparámetros...")
            search.fit(self.X_train, self.y_train)
            logger.info("Búsqueda de hiperparámetros completada.")
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

            datasource_dict = {
                "name": "seoul_bike_sharing_featured",
                "source": str(self.config["dataset_path"]),
                "format": "csv",
                "type": "tabular",
                "description": "Processed dataset with engineered features for bike sharing demand",
                "tags": {"project": "Seoul Bike Sharing MLOps", "stage": "feature_engineering"},
                "size_rows": len(self.X_train) + len(self.X_test),
                "size_columns": len(self.X_train.columns),
            }
            datasource = DatasetSource.from_dict(datasource_dict)
            print("TYPE OF DATASOURCE:", type(datasource))
            best_model = search.best_estimator_
            best_params = search.best_params_
            best_cv_score = search.best_score_

            # Log CV metrics
            mlflow.log_metrics({"cv_best_score": best_cv_score})
            mlflow.log_params(best_params)

            # Evaluate on test set
            y_pred = best_model.predict(self.X_test)
            test_r2 = r2_score(self.y_test, y_pred)
            test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))

            mlflow.log_metrics({"test_r2": test_r2, "test_rmse": test_rmse})
            logger.info(f"Resultados Test - R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}")

            # Save model
            model_path = self.config["model_path"]
            model_path.parent.mkdir(parents=True, exist_ok=True)
            with open(model_path, "wb") as f:
                pickle.dump(best_model, f)
            mlflow.log_artifact(str(model_path))
            mlflow.sklearn.log_model(best_model, artifact_path="model")

            logger.success(
                f"Modelo '{self.config['model_name']}' guardado y registrado en MLflow."
            )


# -------------------------------------------------------------------
# Main training pipeline
# -------------------------------------------------------------------
@app.command()
def main(
    dataset_path: Path = PROCESSED_DATA_DIR / "seoul_bike_sharing_featured.csv",
    target_col: str = TARGET_COL,
    model_name: str = DEFAULT_MODEL_NAME,
    param_grid: str = json.dumps(DEFAULT_PARAM_GRID),
    metric: str = DEFAULT_METRIC,
    search_mode: str = DEFAULT_SEARCH_MODE,
    search_params: str = json.dumps(DEFAULT_SEARCH_PARAMS),
    cv: int = DEFAULT_CV,
    test_size: float = 0.2,
    model_path: Path = DEFAULT_MODEL_PATH,
):
    """Entrena, tunea y evalúa un modelo usando MLflow + DagsHub."""
    training_config = {
        "dataset_path": dataset_path,
        "target_col": target_col,
        "model_name": model_name,
        "param_grid": json.loads(param_grid),
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
