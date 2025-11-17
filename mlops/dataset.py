"""
Module for the initial cleaning and preprocessing of the Seoul Bike Sharing dataset.
Functions:
- Loads the raw dataset.
- Normalizes column names.
- Converts the date column to datetime format.
- Filters out non-operational days.
- Saves the processed dataset in the `processed` folder with MLflow run logging via DagsHub.
"""

from pathlib import Path
from typing import Optional
import os
import dagshub
import pandas as pd
import mlflow
import mlflow.data.pandas_dataset
from loguru import logger
import typer

from mlops.config import INTERIM_DATA_DIR, RAW_DATA_DIR, TARGET_COL, setup_mlflow_connection

app = typer.Typer()


class DatasetProcessor:
    """
    Encapsulates the data preprocessing pipeline.

    This class organizes the loading, cleaning, transformation, and saving of data,
    making the process more maintainable and testable.
    """

    def __init__(self, input_path: Path, output_path: Path):
        self.input_path = input_path
        self.output_path = output_path
        self.df: Optional[pd.DataFrame] = None

    def load_data(self) -> "DatasetProcessor":
        """Loads the dataset from the `input_path`."""
        logger.info(f"Loading dataset from {self.input_path}...")
        try:
            self.df = pd.read_csv(self.input_path, encoding="cp1252")
        except FileNotFoundError:
            logger.error(f"File not found at path: {self.input_path}")
            raise typer.Exit(code=1)
        return self
    
    
    def Load_from_dataframe(self, df:pd.DataFrame) -> "DatasetProcessor":
        """creates the input df from the provided dictionary."""
        logger.info(f"Loading dataset from dataframe...")
        self.df = df
        return self
    
    def to_df(self) -> pd.DataFrame:
        """Export processed data as a DataFrame."""
        if self.df is None:
            raise ValueError("No data available.")
        logger.success("Processing completed successfully.")
        return self.df

    def clean_data_values(self) -> "DatasetProcessor":
        """Cleans the values of 'object' type columns."""
        if self.df is None:
            raise ValueError("DataFrame has not been loaded.")
        logger.info("Cleaning text values in columns...")
        for col in self.df.select_dtypes(include=["object"]).columns:
            # Remove leading and trailing whitespace
            self.df[col] = self.df[col].str.strip()
        # Impute null values in the 'hour' column with the median
        if "hour" in self.df.columns:
            self.df["hour"] = self.df["hour"].fillna(self.df["hour"].median())

        # Drop rows where the target column has null values
        if TARGET_COL in self.df.columns:
            initial_count = len(self.df)
            self.df = self.df.dropna(subset=[TARGET_COL])
            dropped_count = initial_count - len(self.df)
            if dropped_count > 0:
                logger.warning(f"Dropped {dropped_count} rows with null values in '{TARGET_COL}' column.")
        return self

    def _normalize_column_names(self) -> "DatasetProcessor":
        """Normalizes column names to a snake_case format."""
        if self.df is None:
            raise ValueError("DataFrame has not been loaded.")
        logger.info("Normalizing column names...")
        self.df.columns = (
            self.df.columns.str.strip()
            .str.lower()
            .str.replace(r"[^a-zA-Z0-9_]+", "_", regex=True)
            .str.replace(r"_+", "_", regex=True)
            .str.strip("_")
        )
        return self

    def preprocess_data(self) -> "DatasetProcessor":
        if self.df is None:
            raise ValueError("DataFrame has not been loaded.")
        logger.info("Applying preprocessing: converting dates and filtering non-functional days.")
        self._normalize_column_names()
        logger.info("Normalization applied")
        self.df["date"] = pd.to_datetime(self.df["date"], dayfirst=True)
        self.df = self.df[self.df["functioning_day"] == "Yes"].copy()
        self.df.rename(
            columns={"rented_bike_count": TARGET_COL.lower().replace(" ", "_")},
            inplace=True,
        )
        return self

    def save_data(self) -> None:
        """Saves the processed DataFrame to the `output_path`."""
        if self.df is None:
            raise ValueError("No data to save.")
        logger.info(f"Saving processed dataset to {self.output_path}...")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(self.output_path, index=False)
        logger.success("Processing and saving completed successfully.")


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "seoul_bike_sharing.csv",
    output_path: Path = INTERIM_DATA_DIR / "seoul_bike_sharing_cleaned.csv",
):
    """Runs the complete data processing pipeline with MLflow logging."""
    setup_mlflow_connection()
    mlflow.set_experiment("data_preprocessing")

    with mlflow.start_run(run_name="dataset_preprocessing"):
        mlflow.log_param("input_path", str(input_path))
        mlflow.log_param("output_path", str(output_path))

        processor = DatasetProcessor(input_path, output_path)
        processor.load_data().clean_data_values().preprocess_data().save_data()

        # Log dataset information
        if processor.df is not None:
            mlflow.log_metric("rows_processed", len(processor.df))
            mlflow.log_metric("columns", len(processor.df.columns))
            mlflow.log_artifact(str(output_path), artifact_path="processed_data")

        logger.success("Run successfully logged to MLflow/DagsHub.")


if __name__ == "__main__":
    app()