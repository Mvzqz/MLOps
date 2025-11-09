#Prueba el entrenamiento y evaluación del modelo (train.py o modeling.py).

import pytest
import pandas as pd
from mlops.modeling.train import ModelTrainer
import tempfile
from pathlib import Path

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'temperature': [10, 20, 15, 25, 30],
        'humidity': [50, 60, 55, 65, 70],
        'rented_bike_count': [100, 200, 150, 300, 400]
    })

def test_split_data(sample_data):
    trainer = ModelTrainer(sample_data, target_col='rented_bike_count')
    X_train, X_test, y_train, y_test = trainer.split_data()
    assert len(X_train) > 0
    assert len(y_test) > 0

def test_train_model(sample_data):
    trainer = ModelTrainer(sample_data, target_col='rented_bike_count')
    trainer.split_data()
    trainer.train_model()
    assert trainer.model is not None

def test_evaluate_model(sample_data):
    trainer = ModelTrainer(sample_data, target_col='rented_bike_count')
    trainer.split_data()
    trainer.train_model()
    metrics = trainer.evaluate_model()
    assert 'r2' in metrics
    assert 'rmse' in metrics

def test_save_model(sample_data):
    trainer = ModelTrainer(sample_data, target_col='rented_bike_count')
    trainer.split_data()
    trainer.train_model()
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / 'test_model.pkl'
        trainer.save_model(model_path)
        assert model_path.exists()