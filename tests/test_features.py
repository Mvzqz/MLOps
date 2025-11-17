"""
Tests for the feature engineering module (mlops/features.py).

These tests verify that the feature engineering functions are applied correctly.
"""

import pytest
import pandas as pd
from mlops.features import create_features

@pytest.fixture
def sample_data():
    """Fixture to create a sample DataFrame for feature engineering tests."""
    # Includes 'holiday' and 'functioning_day' as they are used in the pipeline.
    return pd.DataFrame({
        'date': pd.to_datetime(['2021-01-01', '2021-01-02']),  # Vie, Sáb
        'hour': [10, 15],
        'temperature': [5.0, 15.0],
        'humidity': [60, 40],
        'holiday': ['No Holiday', 'Holiday'],
        'functioning_day': ['Yes', 'Yes'],
    })

def test_add_time_features(sample_data):
    """Tests the creation of time-based features."""
    df = create_features(sample_data)

    # Check that temporal columns are created
    for col in ["year", "month", "day", "dayofweek", "is_weekend"]:
        assert col in df.columns

    # Simple check: 2021-01-01 is a Friday (not weekend), 2021-01-02 is a Saturday (weekend)
    assert df.loc[0, "is_weekend"] == 0
    assert df.loc[1, "is_weekend"] == 1

    # 'hour' should remain available for cyclical features
    assert "hour" in df.columns

def test_add_interaction_features(sample_data):
    """Tests the creation of interaction and business-logic features."""
    df = create_features(sample_data)

    # Check for interaction columns from the code
    assert "is_rush_hour" in df.columns
    assert "is_holiday_or_weekend" in df.columns

    # Hours 10 and 15 are not rush hour (defined as 7,8,9,17,18,19)
    assert df["is_rush_hour"].tolist() == [0, 0]

    # Row 0: Friday without holiday -> 0; Row 1: Saturday and holiday -> 1
    assert df["is_holiday_or_weekend"].tolist() == [0, 1]

def test_prepare_final_dataset_like(sample_data):
    """
    Tests the final state of the dataset, ensuring it's ready for modeling.
    This includes checking for cyclical features.
    """
    df = create_features(sample_data)

    # The pipeline should leave a non-empty DataFrame with cyclical features
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    for col in ["hour_sin", "hour_cos", "month_sin", "month_cos"]:
        assert col in df.columns
