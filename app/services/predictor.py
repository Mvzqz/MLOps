# app/services/predictor.py
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict
from app.schemas.predict import PredictionRequest
from loguru import logger
import pickle


class Predictor:
    """Handles model loading and prediction logic."""

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.model= self.load_model()


    def predict(self, data: PredictionRequest) -> Dict[str, float]:
        """Perform prediction and return a mock result."""

        processed_data = self._preprocess(data.features)
        prediction = self.model.predict(processed_data)
        final_output = self._postprocess(prediction)
        return {"prediction": final_output}
    
    def _preprocess(self, features):
        """
        Preprocess the input data before making predictions.

        Args:
            input_data (any): The raw input data.

        Returns:
            any: The preprocessed data.
        """
        logger.info("preprocessing input data...")
        X = pd.DataFrame([features])
        X.columns = ["hour","temperature_c","humidity","wind_speed_m_s","visibility_10m","dew_point_temperature_c","solar_radiation_mj_m2","rainfall_mm","snowfall_cm","seasons","mixed_type_col","year","month","day","dayofweek","is_weekend","hour_sin","hour_cos","month_sin","month_cos","is_rush_hour","is_holiday_or_weekend"]
        return X

    def _postprocess(self, prediction):
        """
        Postprocess the prediction after obtaining it from the model.

        Args:
            prediction (any): The raw prediction from the model.

        Returns:
            any: The postprocessed prediction.
        """
        logger.info("postprocessing prediction...")
        return prediction  # Placeholder
    
    def load_model(self):
        """Carga el modelo entrenado desde el `model_path`."""
        logger.info(f"Cargando modelo desde {self.model_path}...")
        try:
            with open(self.model_path, "rb") as f:
                model = pickle.load(f)
        except FileNotFoundError:
            logger.error(f"El modelo no se encontró en: {self.model_path}")
        return model
    
