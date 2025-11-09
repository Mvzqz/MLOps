#Este prueba el módulo features.py, 
#verificando que la ingeniería de características se aplique correctamente.

import pytest
import pandas as pd
from mlops.features import FeatureEngineer

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'date': pd.to_datetime(['2021-01-01', '2021-01-02']),
        'hour': [10, 15],
        'temperature': [5.0, 15.0],
        'humidity': [60, 40]
    })

def test_add_time_features(sample_data):
    fe = FeatureEngineer(sample_data)
    fe.add_time_features()
    assert 'day' in fe.df.columns
    assert 'month' in fe.df.columns
    assert 'hour' in fe.df.columns

def test_add_interaction_features(sample_data):
    fe = FeatureEngineer(sample_data)
    fe.add_interaction_features()
    assert 'temp_humidity_ratio' in fe.df.columns
    assert all(fe.df['temp_humidity_ratio'] == fe.df['temperature'] / fe.df['humidity'])

def test_prepare_final_dataset(sample_data):
    fe = FeatureEngineer(sample_data)
    fe.add_time_features()
    fe.add_interaction_features()
    final_df = fe.prepare_final_dataset()
    assert isinstance(final_df, pd.DataFrame)
    assert not final_df.empty