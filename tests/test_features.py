# Prueban mlops/features.py verificando que la ingeniería de características se aplique correctamente
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import pandas as pd
from pathlib import Path
from mlops.features import FeatureEngineer

@pytest.fixture
def sample_data():
    # Incluimos 'holiday' y 'functioning_day' porque el pipeline los usa
    return pd.DataFrame({
        'date': pd.to_datetime(['2021-01-01', '2021-01-02']),  # Vie, Sáb
        'hour': [10, 15],
        'temperature': [5.0, 15.0],
        'humidity': [60, 40],
        'holiday': ['No Holiday', 'Holiday'],
        'functioning_day': ['Yes', 'Yes'],
    })

def _run_pipeline_with_tmp_paths(df: pd.DataFrame, tmp_path: Path) -> pd.DataFrame:
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    df.to_csv(input_path, index=False)

    fe = FeatureEngineer(input_path=input_path, output_path=output_path)
    fe.load_data().create_features()
    assert fe.df is not None
    return fe.df

def test_add_time_features(sample_data, tmp_path):
    df = _run_pipeline_with_tmp_paths(sample_data, tmp_path)

    # Columnas temporales creadas por _add_temporal_features
    for col in ["year", "month", "day", "dayofweek", "is_weekend"]:
        assert col in df.columns

    # Chequeo simple: 2021-01-01 es viernes (no fin de semana), 2021-01-02 es sábado (fin de semana)
    assert df.loc[0, "is_weekend"] == 0
    assert df.loc[1, "is_weekend"] == 1

    # 'hour' permanece disponible y se usa para cíclicas
    assert "hour" in df.columns

def test_add_interaction_features(sample_data, tmp_path):
    df = _run_pipeline_with_tmp_paths(sample_data, tmp_path)

    # Columnas de interacción reales del código
    assert "is_rush_hour" in df.columns
    assert "is_holiday_or_weekend" in df.columns

    # Horas 10 y 15 no son rush hour (7,8,9,17,18,19)
    assert df["is_rush_hour"].tolist() == [0, 0]

    # Fila 0: viernes sin holiday -> 0; Fila 1: sábado y holiday -> 1
    assert df["is_holiday_or_weekend"].tolist() == [0, 1]

def test_prepare_final_dataset_like(sample_data, tmp_path):
    df = _run_pipeline_with_tmp_paths(sample_data, tmp_path)

    # El pipeline deja un DataFrame no vacío y con features cíclicas
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    for col in ["hour_sin", "hour_cos", "month_sin", "month_cos"]:
        assert col in df.columns
