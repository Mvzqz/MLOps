# tests/test_modeling.py
# Prueba el entrenamiento y evaluación del modelo (mlops/modeling/train.py)

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import pickle
import pytest
import pandas as pd
from pathlib import Path
from mlops.modeling.train import ModelTrainer

@pytest.fixture
def sample_data():
    # Incluimos 'year' porque el trainer ordena por 'year' y divide temporalmente
    return pd.DataFrame({
        'year': [2018, 2019, 2020, 2021, 2022],
        'temperature': [10, 20, 15, 25, 30],
        'humidity': [50, 60, 55, 65, 70],
        'rented_bike_count': [100, 200, 150, 300, 400],
    })

def make_config(tmp_path: Path, df: pd.DataFrame) -> dict:
    dataset_path = tmp_path / "featured.csv"
    df.to_csv(dataset_path, index=False)

    model_path = tmp_path / "model.pkl"
    # Usamos un grid mínimo para el modelo por defecto
    param_grid = {"model__max_depth": [3]}

    return {
        "dataset_path": dataset_path,
        "target_col": "rented_bike_count",
        "model_name": "hist_gradient_boosting_regressor",
        "param_grid": param_grid,
        "metric": "r2",
        "search_mode": "grid",          # evita la búsqueda halving
        "search_params": {},            # no relevantes para 'grid'
        "cv": 2,                        # válido para 5 filas con TimeSeriesSplit
        "test_size": 0.4,               # 3/2 split temporal
        "model_path": model_path,
    }

def test_split_data(sample_data, tmp_path):
    cfg = make_config(tmp_path, sample_data)
    trainer = ModelTrainer(cfg)
    trainer._load_and_split_data()

    assert trainer.X_train is not None and len(trainer.X_train) > 0
    assert trainer.X_test  is not None and len(trainer.X_test)  > 0
    assert trainer.y_train is not None and len(trainer.y_train) > 0
    assert trainer.y_test  is not None and len(trainer.y_test)  > 0

def test_train_and_save_model(sample_data, tmp_path):
    cfg = make_config(tmp_path, sample_data)
    trainer = ModelTrainer(cfg)
    trainer.run()

    # Se guardó el modelo en disco
    assert cfg["model_path"].exists()

def test_infer_on_test_after_run(sample_data, tmp_path):
    cfg = make_config(tmp_path, sample_data)
    trainer = ModelTrainer(cfg)
    trainer.run()

    # Cargamos el modelo guardado y predecimos sobre el test del trainer
    with open(cfg["model_path"], "rb") as f:
        model = pickle.load(f)

    assert trainer.X_test is not None and len(trainer.X_test) > 0
    y_pred = model.predict(trainer.X_test)
    assert len(y_pred) == len(trainer.X_test)

def test_model_artifact_path_created(sample_data, tmp_path):
    cfg = make_config(tmp_path, sample_data)
    trainer = ModelTrainer(cfg)
    trainer.run()

    # El directorio del modelo existe (por si usas subcarpetas)
    assert cfg["model_path"].parent.exists()
