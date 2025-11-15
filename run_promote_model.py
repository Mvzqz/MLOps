"""
Este script consulta los experimentos de MLflow, identifica el mejor modelo
según una métrica y lo guarda en la carpeta 'models/' para que DVC lo rastree.
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
    Encuentra el mejor modelo en un experimento de MLflow y lo guarda localmente.

    Args:
        experiment_name (str): Nombre del experimento en MLflow.
        metric (str): Métrica a usar para determinar el mejor modelo (menor valor).
        model_path (Path): Ruta donde se guardará el mejor modelo.
    """
    setup_mlflow_connection()
    logger.info(f"Buscando el mejor modelo en el experimento '{experiment_name}'...")

    # Obtener el ID del experimento
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        logger.error(f"Experimento '{experiment_name}' no encontrado.")
        return

    # Buscar la mejor corrida (run)
    best_run = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} ASC"],
        max_results=1,
    ).iloc[0]

    logger.info(f"Mejor corrida encontrada: {best_run.run_id} con {metric}: {best_run[f'metrics.{metric}']:.4f}")

    # Descargar y guardar el modelo
    model_uri = f"runs:/{best_run.run_id}/model"
    best_model = mlflow.sklearn.load_model(model_uri)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)

    logger.success(f"Mejor modelo guardado en: {model_path}")


if __name__ == "__main__":
    # El nombre del experimento debe coincidir con el usado en train.py
    promote_best_model(experiment_name="bike_demand_prediction")