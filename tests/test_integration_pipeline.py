"""
Pruebas de integración del pipeline MLOps.
Valida el flujo completo: carga de datos → preprocesamiento → ingeniería → entrenamiento → predicción → evaluación.
"""

import pytest
import pandas as pd
from pathlib import Path
from mlops.dataset import DatasetProcessor
from mlops.features import crear_pipeline_preprocesamiento
from mlops.modeling.train import train_and_evaluate_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import tempfile

@pytest.fixture
def sample_dataset():
    """Genera un dataset temporal simulado similar al original."""
    df = pd.DataFrame({
        'Date': ['01/11/2017', '02/11/2017', '03/11/2017', '04/11/2017'],
        'Functioning Day': ['Yes', 'Yes', 'Yes', 'No'],
        'Hour': [1, 2, 3, 4],
        'Temperature(°C)': [10, 15, 20, 25],
        'Humidity(%)': [50, 55, 60, 65],
        'Visibility (10m)': [1500, 1800, 1200, 1600],
        'Rented Bike Count': [100, 150, 200, 250]
    })
    return df


def test_full_pipeline_integration(sample_dataset):
    """Prueba de integración completa del pipeline."""

    # --- Guardar dataset simulado en archivo temporal ---
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "raw.csv"
        output_path = Path(tmpdir) / "processed.csv"
        sample_dataset.to_csv(input_path, index=False)

        # --- Carga y limpieza de datos ---
        processor = DatasetProcessor(input_path, output_path)
        processor.load_data()
        processor.clean_data_values()
        processor.preprocess_data()
        processor.save_data()

        assert output_path.exists(), " No se generó el dataset limpio"
        df_clean = pd.read_csv(output_path)
        assert not df_clean.isnull().any().any(), "Aún existen valores nulos"

        # --- Ingeniería de características ---
        vars_numericas = ['Hour', 'Temperature(°C)', 'Humidity(%)', 'Visibility (10m)']
        vars_categoricas = []  # se pueden agregar más columnas categóricas
        preprocessor = crear_pipeline_preprocesamiento(vars_numericas, vars_categoricas)

        X = df_clean[vars_numericas]
        y = df_clean['Rented Bike Count']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        preprocessor.fit(X_train)
        X_train_t = preprocessor.transform(X_train)
        X_test_t = preprocessor.transform(X_test)

        assert X_train_t.shape[1] == len(vars_numericas), "Error en número de columnas procesadas"

        # --- Entrenamiento del modelo ---
        model = LinearRegression()
        model.fit(X_train_t, y_train)

        y_pred = model.predict(X_test_t)
        score = r2_score(y_test, y_pred)

        print(f"Score R2 del modelo: {score:.3f}")
        assert score > 0, " El modelo no logró generalizar correctamente"

        # --- Serialización temporal (simula save_model) ---
        model_path = Path(tmpdir) / "temp_model.pkl"
        import joblib
        joblib.dump(model, model_path)
        assert model_path.exists(), "No se guardó el modelo correctamente"

        # --- Carga y predicción ---
        model_loaded = joblib.load(model_path)
        y_pred_loaded = model_loaded.predict(X_test_t)
        assert all(abs(y_pred - y_pred_loaded) < 1e-6), "Inconsistencia entre predicciones guardadas y cargadas"

        print(" Pipeline de integración completado con éxit")