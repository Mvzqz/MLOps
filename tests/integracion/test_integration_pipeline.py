import pytest
from pathlib import Path
import pandas as pd

# Import the function directly, not a class
from mlops.features import create_features

@pytest.fixture
def sample_data(tmp_path):
    """Crea un archivo CSV temporal con datos de ejemplo para pruebas."""
    data = {
        "Date": ["2018-12-01", "2018-12-02"],
        "Hour": [10, 15],
        "temperature_c": [5.3, 8.1],
        "humidity": [37, 44],
        "Holiday": ["No Holiday", "Holiday"],
        "functioning_day": ["Yes", "Yes"],
        "rented_bike_count": [100, 150], # Add target column for lag/rolling features
    }
    df = pd.DataFrame(data)
    input_file = tmp_path / "seoul_bike_sharing_cleaned.csv"
    df.to_csv(input_file, index=False)
    return df


def test_pipeline_feature_engineering(tmp_path, sample_data):
    """Verifica que el pipeline de FeatureEngineer genera un archivo procesado."""
    output_file = tmp_path / "seoul_bike_sharing_featured.csv"

    # Call the feature engineering function directly
    featured_df = create_features(sample_data)

    # Save the result to validate
    featured_df.to_csv(output_file, index=False)

    #Validaciones
    assert output_file.exists(), "El archivo procesado no fue generado."

    df = pd.read_csv(output_file)
    assert not df.empty, "El archivo de salida está vacío."
    assert "hour_sin" in df.columns, "No se generaron las características cíclicas."
    assert "is_rush_hour" in df.columns, "No se generaron las características de negocio."