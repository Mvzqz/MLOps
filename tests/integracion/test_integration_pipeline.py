import sys, os
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "...")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..")))
import pytest
from pathlib import Path
import pandas as pd

from mlops.features import FeatureEngineer  #usamos la clase directamente


@pytest.fixture
def sample_data(tmp_path):
    """Crea un archivo CSV temporal con datos de ejemplo para pruebas."""
    data = {
        "Date": ["2018-12-01", "2018-12-02"],
        "Hour": [10, 15],
        "Temperature(°C)": [5.3, 8.1],
        "Humidity(%)": [37, 44],
        "Holiday": ["No Holiday", "Holiday"],
        "Functioning Day": ["Yes", "Yes"],
    }
    df = pd.DataFrame(data)
    input_file = tmp_path / "seoul_bike_sharing_cleaned.csv"
    df.to_csv(input_file, index=False)
    return input_file


def test_pipeline_feature_engineering(tmp_path, sample_data):
    """Verifica que el pipeline de FeatureEngineer genera un archivo procesado."""
    output_file = tmp_path / "seoul_bike_sharing_featured.csv"

    #Instanciamos y ejecutamos el pipeline
    pipeline = FeatureEngineer(sample_data, output_file)
    pipeline.load_data().create_features().save_data()

    #Validaciones
    assert output_file.exists(), "El archivo procesado no fue generado."

    df = pd.read_csv(output_file)
    assert not df.empty, "El archivo de salida está vacío."
    assert "hour_sin" in df.columns, "No se generaron las características cíclicas."
    assert "is_rush_hour" in df.columns, "No se generaron las características de negocio."