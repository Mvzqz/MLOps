"""Prediction module for the MLOps project."""

from pathlib import Path
import pickle
import numpy as np
from typing import Optional

from loguru import logger
import pandas as pd
import typer
from sklearn.metrics import mean_squared_error, r2_score
from mlops.config import DEFAULT_MODEL_PATH, PROCESSED_DATA_DIR, TARGET_COL

app = typer.Typer()


class ModelPredictor:
    """Encapsulates the logic for loading a model and making predictions."""

    def __init__(self, model_path: Path, features_path: Path, predictions_path: Path):
        self.model_path = model_path
        self.features_path = features_path
        self.predictions_path = predictions_path
        self.model = None
        self.df: Optional[pd.DataFrame] = None
        self.predictions: Optional[pd.Series] = None
        self.log_predictions: Optional[np.ndarray] = None

    def load_model(self) -> "ModelPredictor":
        """Loads the trained model from the `model_path`."""
        logger.info(f"Loading model from {self.model_path}...")
        try:
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
        except FileNotFoundError:
            logger.error(f"Model not found at: {self.model_path}")
            raise typer.Exit(code=1)
        return self

    def load_data(self) -> "ModelPredictor":
        """Loads the data for prediction from `features_path`."""
        logger.info(f"Loading data for prediction from {self.features_path}...")
        try:
            self.df = pd.read_csv(self.features_path)
        except FileNotFoundError:
            logger.error(f"Features file not found at: {self.features_path}")
            raise typer.Exit(code=1)
        return self
    
    def load_data_from_dataframe(self, dataframe: pd.DataFrame) -> "ModelPredictor":
        """Loads the data for prediction from a DataFrame."""
        logger.info("Loading data for prediction from provided DataFrame...")
        self.df = dataframe
        return self

    def predict(self) -> "ModelPredictor":
        """Makes predictions on the loaded data."""
        if self.model is None or self.df is None:
            raise ValueError("Model and data must be loaded before predicting.")

        logger.info("Making predictions...")
        X = self.df.drop(columns=[TARGET_COL], errors="ignore")
        # The model predicts on a logarithmic scale, so we must apply the inverse transformation.
        log_predictions = self.model.predict(X)
        logger.info("Applying inverse transformation (expm1) to predictions.")
        self.log_predictions = log_predictions
        return self

    def post_process(self)-> "ModelPredictor":
        """
        Postprocess the prediction after obtaining it from the model.   

        Args:
            prediction (any): The raw prediction from the model.

        Returns:
            any: The postprocessed prediction.
        """
        logger.info("postprocessing prediction...")
        final_prediction = np.expm1(self.log_predictions)
        self.predictions = final_prediction
        return self

    
    def to_df(self) -> pd.DataFrame:
        """Export processed data as a DataFrame."""
        if self.predictions is None:
            raise ValueError("No predictions available.")
        logger.success("Processing completed successfully.")
        predictions_df = pd.DataFrame({"y_pred": self.predictions})
        return predictions_df

@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "seoul_bike_sharing_featured.csv",
    model_path: Path = DEFAULT_MODEL_PATH,
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
):
    """Runs the complete prediction pipeline."""
    predictor = ModelPredictor(model_path, features_path, predictions_path)
    predictor.load_model().load_data().predict().post_process()

    # Save the predictions
    predictions_df = predictor.to_df()
    if TARGET_COL in predictor.df.columns:
        predictions_df["y_real"] = predictor.df[TARGET_COL]

    predictor.df["y_pred"] = predictions_df["y_pred"] 

    predictions_df.dropna(inplace=True)

    test_r2 = r2_score(predictions_df["y_real"], predictions_df["y_pred"])
    test_rmse = np.sqrt(mean_squared_error(predictions_df["y_real"], predictions_df["y_pred"]))
    logger.info(f"Test R2 Score: {test_r2:.4f}")
    logger.info(f"Test RMSE: {test_rmse:.4f}")

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(predictions_path, index=False)
    logger.success(f"Predictions saved to {predictions_path}")


if __name__ == "__main__":
    app()
