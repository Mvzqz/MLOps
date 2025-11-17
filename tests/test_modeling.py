"""
Tests for the model training and evaluation module (mlops/modeling/train.py).
"""

import json
import pickle
import pytest
import pandas as pd
from pathlib import Path
from mlops.modeling.train import ModelTrainer

@pytest.fixture
def sample_data():
    """
    Provides a sample DataFrame for model training tests, including categorical features.
    """
    return pd.DataFrame({
        'year': [2018, 2019, 2020, 2021, 2022],
        'temperature': [10, 20, 15, 25, 30],
        'humidity': [50, 60, 55, 65, 70],
        'season': ['Winter', 'Spring', 'Summer', 'Autumn', 'Winter'],
        'rented_bike_count': [100, 200, 150, 300, 400],
    })

def make_config(tmp_path: Path, df: pd.DataFrame) -> dict:
    """
    Helper function to create a configuration dictionary for the ModelTrainer.

    Args:
        tmp_path (Path): The temporary directory to store artifacts.
        df (pd.DataFrame): The sample data to be saved as a CSV.

    Returns:
        dict: A configuration dictionary for testing.
    """
    dataset_path = tmp_path / "featured.csv"
    df.to_csv(dataset_path, index=False)

    model_path = tmp_path / "model.pkl"
    param_grid = {"model__max_depth": [3]}

    return {
        "dataset_path": dataset_path,
        "target_col": "rented_bike_count",
        "model_name": "hist_gradient_boosting_regressor",
        "param_grid": param_grid,
        "metric": "r2",
        "search_mode": "grid",
        "search_params": {},
        "cv": 2,
        "test_size": 0.4,
        "model_path": model_path,
        "run_name": "test_run",
        "experiment_id": "0",
    }

def test_split_data(sample_data, tmp_path):
    """Tests the chronological data splitting logic of the ModelTrainer."""
    cfg = make_config(tmp_path, sample_data)
    trainer = ModelTrainer(cfg)
    trainer._load_and_split_data()

    assert trainer.X_train is not None and len(trainer.X_train) > 0
    assert trainer.X_test  is not None and len(trainer.X_test)  > 0
    assert trainer.y_train is not None and len(trainer.y_train) > 0
    assert trainer.y_test  is not None and len(trainer.y_test)  > 0

def test_train_and_save_model(sample_data, tmp_path):
    """
    Tests the full `run` method of the ModelTrainer, ensuring a model is trained
    and saved to disk.
    """
    cfg = make_config(tmp_path, sample_data)
    trainer = ModelTrainer(cfg)
    trainer.run()

    # Check if the model was saved to disk
    assert cfg["model_path"].exists()

def test_infer_on_test_after_run(sample_data, tmp_path):
    """
    Tests that a trained model can be loaded and used for inference on the test set.
    """
    cfg = make_config(tmp_path, sample_data)
    trainer = ModelTrainer(cfg)
    trainer.run()

    # Load the saved model and predict on the trainer's test set
    with open(cfg["model_path"], "rb") as f:
        model = pickle.load(f)

    assert trainer.X_test is not None and len(trainer.X_test) > 0
    y_pred = model.predict(trainer.X_test)
    assert len(y_pred) == len(trainer.X_test)

def test_model_artifact_path_created(sample_data, tmp_path):
    """
    Ensures that the directory for the model artifact is created during the
    training run.
    """
    cfg = make_config(tmp_path, sample_data)
    trainer = ModelTrainer(cfg)
    trainer.run()

    # The model's parent directory should exist (in case of subfolders)
    assert cfg["model_path"].parent.exists()
