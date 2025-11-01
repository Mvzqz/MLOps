"""Training module for the MLOps project."""

import json
from pathlib import Path
import pickle
from typing import Any, Dict, Optional, Tuple

import numpy as np
from loguru import logger
import pandas as pd
from xgboost import XGBRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import (
    GridSearchCV,
    HalvingGridSearchCV,
    HalvingRandomSearchCV,
    TimeSeriesSplit,
)
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
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
    RANDOM_SEED,
    TARGET_COL,
)

app = typer.Typer()

# -------------------------------------------------------------------
# MODEL AND SEARCH REGISTRIES
# -------------------------------------------------------------------
MODEL_REGISTRY = {
    "svr": SVR,
    "hist_gradient_boosting_regressor": HistGradientBoostingRegressor,
    "random_forest_regressor": RandomForestRegressor,
    "xgb_regressor": XGBRegressor,
    # Agrega aquí otros modelos de regresión o clasificación
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

    def _load_and_split_data(self) -> "ModelTrainer":
        """Carga los datos y los divide temporalmente."""
        path = self.config["dataset_path"]
        logger.info(f"Cargando dataset desde {path}...")
        df = pd.read_csv(path).sort_values("year").reset_index(drop=True)

        target_col = self.config["target_col"]
        if target_col not in df.columns:
            logger.error(f"Columna objetivo '{target_col}' no encontrada.")
            raise ValueError(f"Columna objetivo '{target_col}' no encontrada en el dataset. Columnas disponibles: {df.columns.tolist()}")

        # Eliminar filas donde la columna objetivo es NaN para evitar errores en el entrenamiento
        initial_rows = len(df)
        df.dropna(subset=[target_col], inplace=True)
        if len(df) < initial_rows:
            logger.warning(f"Se eliminaron {initial_rows - len(df)} filas con valores nulos en la columna objetivo '{target_col}'.")


        X = df.drop(columns=[target_col])
        y = df[target_col]

        # División temporal para evitar fuga de datos
        split_index = int(len(df) * (1 - self.config["test_size"]))
        self.X_train, self.X_test = X.iloc[:split_index], X.iloc[split_index:]
        self.y_train, self.y_test = y.iloc[:split_index], y.iloc[split_index:]

        logger.info(
            f"División temporal: Train={len(self.X_train)} ({self.X_train['year'].min()}-{self.X_train['year'].max()}), "
            f"Test={len(self.X_test)} ({self.X_test['year'].min()}-{self.X_test['year'].max()})"
        )
        return self

    def _build_pipeline(self) -> Pipeline:
        """Construye el pipeline de preprocesamiento y modelo."""
        model_name = self.config["model_name"]
        if model_name not in MODEL_REGISTRY:
            logger.error(f"Modelo '{model_name}' no encontrado. Disponibles: {list(MODEL_REGISTRY.keys())}")
            raise typer.Exit(code=1)
        
        model_instance = MODEL_REGISTRY[model_name]()
        
        cat_cols = self.X_train.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = self.X_train.select_dtypes(include=np.number).columns.tolist()

        logger.info(f"Columnas categóricas: {cat_cols}")
        logger.info(f"Columnas numéricas: {num_cols}")

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", SimpleImputer(strategy="median"), num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ],
            remainder="drop",
        )

        return Pipeline(steps=[("preprocessor", preprocessor), ("model", model_instance)])

    def _build_search_strategy(self, pipeline: Pipeline):
        """Configura la estrategia de búsqueda de hiperparámetros."""
        search_mode = self.config["search_mode"]
        if search_mode not in SEARCH_REGISTRY:
            logger.error(f"Modo de búsqueda '{search_mode}' no soportado. Elige entre {list(SEARCH_REGISTRY.keys())}")
            raise typer.Exit(code=1)

        time_series_cv = TimeSeriesSplit(n_splits=self.config["cv"])
        search_class = SEARCH_REGISTRY[search_mode]
        
        logger.info(f"Inicializando estrategia de búsqueda: {search_class.__name__}")
        return search_class(
            estimator=pipeline,
            param_grid=self.config["param_grid"],
            scoring=self.config["metric"],
            cv=time_series_cv,
            n_jobs=-1,
            verbose=1,
            **self.config["search_params"],
        )

    def run(self) -> None:
        """Ejecuta el pipeline completo de entrenamiento."""
        self._load_and_split_data()
        pipeline = self._build_pipeline()
        search = self._build_search_strategy(pipeline)

        logger.info(f"Iniciando búsqueda de hiperparámetros para {self.config['model_name']}...")
        search.fit(self.X_train, self.y_train)

        best_model = search.best_estimator_
        self._evaluate_and_save(best_model, search.best_params_, search.best_score_)

    def _evaluate_and_save(self, model: Pipeline, best_params: Dict, best_cv_score: float):
        """Evalúa el modelo en el conjunto de test y lo guarda."""
        y_pred = model.predict(self.X_test)
        test_r2 = r2_score(self.y_test, y_pred)
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        test_results = {"r2": test_r2, "rmse": test_rmse}

        logger.info(f"Resultados en Test: {test_results}")

        model_path = self.config["model_path"]
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        logger.success(f"Mejor modelo guardado en {model_path}")

        self._print_summary(best_params, best_cv_score, test_results)

    def _print_summary(self, best_params: Dict, best_cv_score: float, test_results: Dict):
        """Imprime un resumen del entrenamiento."""
        typer.echo("\n" + "="*25 + " RESUMEN DE ENTRENAMIENTO " + "="*25)
        typer.echo(f"Modelo: {self.config['model_name']}")
        typer.echo(f"Métrica de tuneo: {self.config['metric']}")
        typer.echo(f"Mejor puntuación en CV (TimeSeriesSplit): {best_cv_score:.4f}")
        typer.echo(f"Mejores Hiperparámetros: {best_params}")
        typer.echo(f"Resultados en Test: R²={test_results['r2']:.4f}, RMSE={test_results['rmse']:.2f}")
        typer.echo("="*73 + "\n")


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
    test_size: float = 0.2,  # Usamos un 20% para el test final temporal
    model_path: Path = DEFAULT_MODEL_PATH,
):
    """Entrena, tunea y evalúa un modelo usando un pipeline robusto."""

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
