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

from mlops.config import PROCESSED_DATA_DIR

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
            self.df = pd.read_csv(self.input_path, parse_dates=["date"])
        except FileNotFoundError:
            logger.error(f"El archivo no se encontró en la ruta: {self.input_path}")
            raise typer.Exit(code=1)
        return self

    def create_features(self) -> "FeatureEngineer":
        """Crea y añade nuevas características al DataFrame."""
        if self.df is None:
            raise ValueError("El DataFrame no ha sido cargado. Llama a `load_data` primero.")
        
        logger.info("Creando nuevas características...")
        self._add_temporal_features()
        self._add_cyclical_features()
        self._add_interaction_features()
        
        # Eliminar columnas que ya no son necesarias después de la ingeniería
        self.df = self.df.drop(columns=["date", "functioning_day", "holiday"], errors="ignore")
        
        return self

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
    input_path: Path = PROCESSED_DATA_DIR / "seoul_bike_sharing_cleaned.csv",
    output_path: Path = PROCESSED_DATA_DIR / "seoul_bike_sharing_featured.csv",
):
    """Ejecuta el pipeline completo de ingeniería de características."""
    engineer = FeatureEngineer(input_path, output_path)
    (engineer.load_data()
     .create_features()
     .save_data())


if __name__ == "__main__":
    app()
