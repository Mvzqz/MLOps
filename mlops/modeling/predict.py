"""Prediction module for the MLOps project."""

from pathlib import Path
import pickle
import numpy as np
from typing import Optional

from loguru import logger
import pandas as pd
import typer

from mlops.config import DEFAULT_MODEL_PATH, PROCESSED_DATA_DIR, TARGET_COL

app = typer.Typer()


class ModelPredictor:
    """Encapsula la lógica para cargar un modelo y realizar predicciones."""

    def __init__(self, model_path: Path, features_path: Path, predictions_path: Path):
        self.model_path = model_path
        self.features_path = features_path
        self.predictions_path = predictions_path
        self.model = None
        self.df: Optional[pd.DataFrame] = None
        self.predictions: Optional[pd.Series] = None

    def load_model(self) -> "ModelPredictor":
        """Carga el modelo entrenado desde el `model_path`."""
        logger.info(f"Cargando modelo desde {self.model_path}...")
        try:
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
        except FileNotFoundError:
            logger.error(f"El modelo no se encontró en: {self.model_path}")
            raise typer.Exit(code=1)
        return self

    def load_data(self) -> "ModelPredictor":
        """Carga los datos para la predicción desde `features_path`."""
        logger.info(f"Cargando datos para predicción desde {self.features_path}...")
        try:
            self.df = pd.read_csv(self.features_path)
        except FileNotFoundError:
            logger.error(f"El archivo de características no se encontró en: {self.features_path}")
            raise typer.Exit(code=1)
        return self

    def predict(self) -> "ModelPredictor":
        """Realiza predicciones sobre los datos cargados."""
        if self.model is None or self.df is None:
            raise ValueError("El modelo y los datos deben cargarse antes de predecir.")

        logger.info("Realizando predicciones...")
        X = self.df.drop(columns=[TARGET_COL], errors="ignore")
        # El modelo predice en escala logarítmica, por lo que debemos aplicar la transformación inversa.
        log_predictions = self.model.predict(X)
        logger.info("Aplicando transformación inversa (expm1) a las predicciones.")
        self.predictions = np.expm1(log_predictions)

        return self


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "seoul_bike_sharing_featured.csv",
    model_path: Path = DEFAULT_MODEL_PATH,
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
):
    """Ejecuta el pipeline completo de predicción."""
    predictor = ModelPredictor(model_path, features_path, predictions_path)
    predictor.load_model().load_data().predict()

    # Guardar las predicciones
    predictions_df = pd.DataFrame({"y_pred": predictor.predictions})
    if TARGET_COL in predictor.df.columns:
        predictions_df["y_real"] = predictor.df[TARGET_COL]

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(predictions_path, index=False)
    logger.success(f"Predicciones guardadas en {predictions_path}")


if __name__ == "__main__":
    app()
