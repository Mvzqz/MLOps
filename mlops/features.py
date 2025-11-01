"""
Módulo para la ingeniería de características del dataset Seoul Bike Sharing.

Este script carga el dataset limpio, crea nuevas características y guarda el
dataset enriquecido. Sigue un enfoque orientado a objetos para mayor
modularidad y mantenibilidad.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
import typer

from mlops.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


class FeatureEngineer:
    """Encapsula el pipeline de ingeniería de características."""

    def __init__(self, input_path: Path, output_path: Path):
        self.input_path = input_path
        self.output_path = output_path
        self.df: Optional[pd.DataFrame] = None

    def load_data(self) -> "FeatureEngineer":
        """Carga el dataset limpio."""
        logger.info(f"Cargando dataset desde {self.input_path}...")
        try:
            self.df = pd.read_csv(self.input_path)
        except FileNotFoundError:
            logger.error(f"El archivo no se encontró en la ruta: {self.input_path}")
            raise typer.Exit(code=1)
        return self

    def create_features(self) -> "FeatureEngineer":
        """Crea y añade nuevas características al DataFrame."""
        if self.df is None:
            raise ValueError("El DataFrame no ha sido cargado. Llama a `load_data` primero.")
        
        self._clean_column_names()
        self.df["date"] = pd.to_datetime(self.df["date"], errors="coerce")
        self._convert_numeric_columns()
        
        logger.info("Creando nuevas características...")

        # Imputar 'hour' antes de usarla para crear nuevas características
        if "hour" in self.df.columns:
            self.df["hour"] = self.df["hour"].fillna(self.df["hour"].median())

        self._add_temporal_features()

        self.df["hour"] = pd.to_numeric(self.df["hour"], errors="coerce")
        self.df["month"] = pd.to_numeric(self.df["month"], errors="coerce")

        self._add_cyclical_features()
        self._add_interaction_features()
        
        # Eliminar columnas que ya no son necesarias después de la ingeniería
        self.df = self.df.drop(columns=["date", "functioning_day", "holiday"], errors="ignore")
        
        return self

    def _clean_column_names(self):
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

    def _convert_numeric_columns(self):
        """
        Convierte columnas que parecen numéricas (pero son 'object') a tipo numérico.
        """
        logger.info("Intentando convertir columnas de tipo 'object' a numérico...")
        for col in self.df.select_dtypes(include=["object"]).columns:
            # Intentar convertir a numérico, ignorando errores para no-numéricos
            # y asegurando que la columna de tipo mixto se mantenga como 'object'
            if col == "mixed_type_col":
                self.df[col] = self.df[col].astype("object")
                continue

            numeric_col = pd.to_numeric(self.df[col], errors="coerce")
            
            # Si una porción significativa de la columna es numérica, la convertimos
            if self.df is not None and numeric_col.notna().sum() / len(self.df[col].dropna()) > 0.8:
                self.df[col] = numeric_col
                logger.info(f"  - Columna '{col}' convertida a numérico.")

    def _add_temporal_features(self):
        """Añade características basadas en la columna 'date'."""
        self.df["year"] = self.df["date"].dt.year
        self.df["month"] = self.df["date"].dt.month
        self.df["day"] = self.df["date"].dt.day
        self.df["dayofweek"] = self.df["date"].dt.dayofweek
        self.df["is_weekend"] = (self.df["dayofweek"] >= 5).astype(int)

    def _add_cyclical_features(self):
        """Añade características cíclicas para entender patrones temporales."""
        self.df["hour_sin"] = np.sin(2 * np.pi * self.df["hour"] / 24.0)
        self.df["hour_cos"] = np.cos(2 * np.pi * self.df["hour"] / 24.0)
        self.df["month_sin"] = np.sin(2 * np.pi * (self.df["month"] - 1) / 12.0)
        self.df["month_cos"] = np.cos(2 * np.pi * (self.df["month"] - 1) / 12.0)

    def _add_interaction_features(self):
        """Añade características de negocio y de interacción."""
        self.df["is_rush_hour"] = self.df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
        self.df["is_holiday_or_weekend"] = ((self.df["is_weekend"] == 1) | (self.df["holiday"] == "Holiday")).astype(int)

    def save_data(self) -> None:
        """Guarda el DataFrame con características en el `output_path`."""
        if self.df is None:
            raise ValueError("No hay datos para guardar.")
        logger.info(f"Guardando dataset con características en {self.output_path}...")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(self.output_path, index=False)
        logger.success("Ingeniería de características y guardado completados.")

@app.command()
def main(
    input_path: Path = INTERIM_DATA_DIR / "seoul_bike_sharing_cleaned.csv",
    output_path: Path = PROCESSED_DATA_DIR / "seoul_bike_sharing_featured.csv",
):
    """Ejecuta el pipeline completo de ingeniería de características."""
    engineer = FeatureEngineer(input_path, output_path)
    (engineer.load_data()
     .create_features()
     .save_data())


if __name__ == "__main__":
    app()
