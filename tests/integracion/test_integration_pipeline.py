import pytest
from pathlib import Path
import pandas as pd

from mlops.features import create_features

@pytest.fixture
def sample_data(tmp_path):
    """
    Creates a sample DataFrame for integration testing.

    Args:
        tmp_path (Path): A temporary directory path provided by pytest.

    Returns:
        pd.DataFrame: A sample DataFrame representing cleaned data.
    """
    data = {
        "date": ["2018-12-01", "2018-12-02"],
        "hour": [10, 15],
        "temperature_c": [5.3, 8.1],
        "humidity": [37, 44],
        "holiday": ["No Holiday", "Holiday"],
        "functioning_day": ["Yes", "Yes"],
        "rented_bike_count": [100, 150],
    }
    df = pd.DataFrame(data)
    input_file = tmp_path / "seoul_bike_sharing_cleaned.csv"
    df.to_csv(input_file, index=False)
    return df


def test_pipeline_feature_engineering(tmp_path, sample_data):
    """
    Tests the feature engineering pipeline from a sample DataFrame to a processed file.

    This test verifies that the `create_features` function correctly processes
    a raw DataFrame and that the output contains the expected new features.

    Args:
        tmp_path (Path): A temporary directory path provided by pytest.
        sample_data (pd.DataFrame): The sample data fixture.
    """
    output_file = tmp_path / "seoul_bike_sharing_featured.csv"

    # Call the feature engineering function directly
    featured_df = create_features(sample_data)

    # Save the result to validate
    featured_df.to_csv(output_file, index=False)

    assert output_file.exists(), "The processed file was not generated."

    df = pd.read_csv(output_file)
    assert not df.empty, "The output file is empty."
    assert "hour_sin" in df.columns, "Cyclical features were not generated."
    assert "is_rush_hour" in df.columns, "Business-related features were not generated."