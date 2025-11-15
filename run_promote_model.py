"""
This script queries MLflow experiments, identifies the best model
based on a metric, and saves it to the 'models/' folder for DVC to track.
"""
import os
import pickle
from pathlib import Path

import mlflow
from loguru import logger

from mlops.config import MODELS_DIR, setup_mlflow_connection


def promote_best_model(
    experiment_name: str = "data_science_experiments",
    metric: str = "test_rmse",
    model_path: Path = MODELS_DIR / "best_model.pkl",
):
    """
    Finds the best model in an MLflow experiment and saves it locally.

    Args:
        experiment_name (str): Name of the experiment in MLflow.
        metric (str): Metric to use to determine the best model (lower is better).
        model_path (Path): Path where the best model will be saved.
    """
    setup_mlflow_connection()
    logger.info(f"Searching for the best model in experiment '{experiment_name}'...")

    # Get the experiment ID
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        logger.error(f"Experiment '{experiment_name}' not found.")
        return

    # Search for the best run
    best_run = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} ASC"],
        max_results=1,
    ).iloc[0]

    logger.info(f"Best run found: {best_run.run_id} with {metric}: {best_run[f'metrics.{metric}']:.4f}")

    # Download and save the model
    model_uri = f"runs:/{best_run.run_id}/model"
    best_model = mlflow.sklearn.load_model(model_uri)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)

    logger.success(f"Best model saved to: {model_path}")


if __name__ == "__main__":
    # The experiment name must match the one used in train.py
    promote_best_model(experiment_name="bike_demand_prediction")