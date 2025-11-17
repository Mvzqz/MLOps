# app/services/predictor.py
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict
from app.schemas.predict import FeatureRow, PredictRequest
from loguru import logger
from mlops.dataset import DatasetProcessor 
from mlops.features import create_features
from mlops.modeling.predict import ModelPredictor


class Predictor:
    """Handles model loading and prediction logic."""

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.model_predictor = ModelPredictor(model_path, None, None)


    def predict(self, data: PredictRequest) -> Dict[str, float]:
        """Perform prediction and return a mock result."""

        processed_data = self._preprocess(data.Features)
        predictions_df = self.model_predictor.load_model().load_data_from_dataframe(processed_data).predict().post_process().to_df()
        final_output = predictions_df["y_pred"].tolist()
        return {"Prediction": final_output}
    
    def _preprocess(self, features: list[FeatureRow]):
        """
        Preprocess the input data before making predictions.

        Args:
            input_data (any): The raw input data.

        Returns:
            any: The preprocessed data.
        """
        logger.info("preprocessing input data...")
        processor = DatasetProcessor("", "")
        rows_as_dicts = [row.model_dump() for row in features]
        feature_df = pd.DataFrame(rows_as_dicts)
        logger.info("Cleaning input data...")
        clean_data = processor.Load_from_dataframe(feature_df).clean_data_values().preprocess_data().to_df()
        logger.info("Creating features...")
        features = create_features(clean_data)
        return features

    def _postprocess(self, prediction):
        """
        Postprocess the prediction after obtaining it from the model.   

        Args:
            prediction (any): The raw prediction from the model.

        Returns:
            any: The postprocessed prediction.
        """
        logger.info("postprocessing prediction...")
        final_prediction = np.expm1(prediction)
        return final_prediction
    

