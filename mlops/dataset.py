"""
Módulo para la limpieza y preprocesamiento inicial del dataset Seoul Bike Sharing,
siguiendo un enfoque estructurado y modular.

Funciones:
- Carga el dataset crudo.
- Normaliza los nombres de las columnas.
- Convierte la columna de fecha a formato datetime.
- Filtra los días no operativos.
- Guarda el dataset procesado en la carpeta `processed`.
"""

from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger
import typer

from mlops.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, TARGET_COL

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
        """Aplica las transformaciones de preprocesamiento."""
        if self.df is None:
            raise ValueError("El DataFrame no ha sido cargado.")
        logger.info("Aplicando preprocesamiento: convirtiendo fechas y filtrando días no funcionales.")
        self._normalize_column_names()
        self.df["date"] = pd.to_datetime(self.df["date"], dayfirst=True)
        self.df = self.df[self.df["functioning_day"] == "Yes"].copy()
        # Renombrar la columna objetivo para que coincida con la configuración
        self.df.rename(columns={"rented_bike_count": TARGET_COL.lower().replace(" ", "_")}, inplace=True)
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
    output_path: Path = PROCESSED_DATA_DIR / "seoul_bike_sharing_cleaned.csv",
):
    """Ejecuta el pipeline completo de procesamiento de datos."""
    processor = DatasetProcessor(input_path, output_path)
    (processor.load_data()
     .clean_data_values()
     .preprocess_data()
     .save_data())


if __name__ == "__main__":
    app()
