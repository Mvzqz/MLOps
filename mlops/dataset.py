"""
Módulo para la limpieza y preprocesamiento inicial del dataset Seoul Bike Sharing,
Funciones:
- Carga el dataset crudo.
- Normaliza los nombres de las columnas.
- Convierte la columna de fecha a formato datetime.
- Filtra los días no operativos.
- Guarda el dataset procesado en la carpeta `processed`.con registro de ejecución en MLflow mediante DagsHub.
"""

from pathlib import Path
from typing import Optional
import os

import pandas as pd
import mlflow
import mlflow.data.pandas_dataset
from loguru import logger
import typer

from mlops.config import INTERIM_DATA_DIR, RAW_DATA_DIR, TARGET_COL

app = typer.Typer()


class DatasetProcessor:
    """
    Encapsula el pipeline de preprocesamiento de datos.

    Esta clase organiza la carga, limpieza, transformación y guardado de los datos,
    haciendo el proceso más mantenible y fácil de probar.
    """

    def __init__(self, input_path: Path, output_path: Path):
        self.input_path = input_path
        self.output_path = output_path
        self.df: Optional[pd.DataFrame] = None

    def load_data(self) -> "DatasetProcessor":
        """Carga el dataset desde el `input_path`."""
        logger.info(f"Cargando dataset desde {self.input_path}...")
        try:
            self.df = pd.read_csv(self.input_path, encoding="cp1252")
        except FileNotFoundError:
            logger.error(f"El archivo no se encontró en la ruta: {self.input_path}")
            raise typer.Exit(code=1)
        return self

    def clean_data_values(self) -> "DatasetProcessor":
        """Limpia los valores de las columnas de tipo 'object'."""
        if self.df is None:
            raise ValueError("El DataFrame no ha sido cargado.")
        logger.info("Limpiando valores de texto en las columnas...")
        for col in self.df.select_dtypes(include=["object"]).columns:
            # Elimina espacios en blanco al inicio y al final
            self.df[col] = self.df[col].str.strip()
        # Imputar valores nulos en la columna 'hour' con la mediana
        if "hour" in self.df.columns:
            self.df["hour"] = self.df["hour"].fillna(self.df["hour"].median())
        return self

    def _normalize_column_names(self) -> "DatasetProcessor":
        """Normaliza los nombres de las columnas a un formato snake_case."""
        if self.df is None:
            raise ValueError("El DataFrame no ha sido cargado.")
        logger.info("Normalizando nombres de columnas...")
        self.df.columns = (
            self.df.columns.str.strip()
            .str.lower()
            .str.replace(r"[^a-zA-Z0-9_]+", "_", regex=True)
            .str.replace(r"_+", "_", regex=True)
            .str.strip("_")
        )
        return self

    def preprocess_data(self) -> "DatasetProcessor":
        if self.df is None:
            raise ValueError("El DataFrame no ha sido cargado.")
        logger.info("Aplicando preprocesamiento: convirtiendo fechas y filtrando días no funcionales.")
        self._normalize_column_names()
        self.df["date"] = pd.to_datetime(self.df["date"], dayfirst=True)
        self.df = self.df[self.df["functioning_day"] == "Yes"].copy()
        self.df.rename(
            columns={"rented_bike_count": TARGET_COL.lower().replace(" ", "_")},
            inplace=True,
        )
        return self

    def save_data(self) -> None:
        """Guarda el DataFrame procesado en el `output_path`."""
        if self.df is None:
            raise ValueError("No hay datos para guardar.")
        logger.info(f"Guardando dataset procesado en {self.output_path}...")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(self.output_path, index=False)
        logger.success("Procesamiento y guardado completados exitosamente.")


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "seoul_bike_sharing.csv",
    output_path: Path = INTERIM_DATA_DIR / "seoul_bike_sharing_cleaned.csv",
):
    """Ejecuta el pipeline completo de procesamiento de datos con logging en MLflow."""
    
    # ---- Configurar MLflow con DagsHub ----
    dagshub_repo = os.getenv("DAGSHUB_REPO")
    dagshub_user = os.getenv("DAGSHUB_USER")

    if dagshub_repo and dagshub_user:
        tracking_uri = f"https://dagshub.com/{dagshub_user}/{dagshub_repo}.mlflow"
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("data_preprocessing")
        logger.info(f"MLflow tracking activo en {tracking_uri}")
    else:
        logger.warning("Variables de entorno de DagsHub no configuradas. Se usará MLflow local.")

    with mlflow.start_run(run_name="dataset_preprocessing"):
        mlflow.log_param("input_path", str(input_path))
        mlflow.log_param("output_path", str(output_path))

        processor = DatasetProcessor(input_path, output_path)
        processor.load_data().clean_data_values().preprocess_data().save_data()

        # Loggear información del dataset
        if processor.df is not None:
            mlflow.log_metric("rows_processed", len(processor.df))
            mlflow.log_metric("columns", len(processor.df.columns))
            mlflow.log_artifact(str(output_path), artifact_path="processed_data")

            # También registrar el dataset como MLflow Dataset
            mlflow.data.log_dataset(
                mlflow.data.pandas_dataset.from_pandas(processor.df, source=str(input_path)),
                context="cleaned_dataset",
            )

        logger.success("Ejecución registrada en MLflow/DagsHub exitosamente.")


if __name__ == "__main__":
    app()
