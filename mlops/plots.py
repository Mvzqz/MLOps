"""Módulo para la generación de gráficos de análisis exploratorio de datos (EDA).

Este script carga los datos procesados y genera visualizaciones clave
para entender la distribución de los datos y las relaciones entre
variables. Sigue un enfoque orientado a objetos para mayor modularidad.
"""

from pathlib import Path
from typing import Optional

from loguru import logger
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import typer

from mlops.config import FIGURES_DIR, PROCESSED_DATA_DIR, TARGET_COL

app = typer.Typer()


class PlotGenerator:
    """Encapsula la lógica para generar y guardar gráficos de EDA."""

    def __init__(self, input_path: Path, output_dir: Path):
        self.input_path = input_path
        self.output_dir = output_dir
        self.df: Optional[pd.DataFrame] = None
        sns.set_theme(style="whitegrid")

    def load_data(self) -> "PlotGenerator":
        """Carga el dataset desde la ruta de entrada."""
        logger.info(f"Cargando datos desde {self.input_path}...")
        try:
            self.df = pd.read_csv(self.input_path)
        except FileNotFoundError:
            logger.error(f"El archivo no se encontró en: {self.input_path}")
            raise typer.Exit(code=1)
        return self

    def generate_plots(self) -> None:
        """Genera y guarda todos los gráficos definidos."""
        if self.df is None:
            raise ValueError("El DataFrame no ha sido cargado. Llama a `load_data` primero.")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Generando gráficos en el directorio: {self.output_dir}")

        self._plot_target_distribution()
        self._plot_demand_by_hour()
        self._plot_correlation_heatmap()

        logger.success("Todos los gráficos han sido generados exitosamente.")

    def _plot_target_distribution(self):
        """Genera y guarda un histograma de la variable objetivo."""
        plt.figure(figsize=(10, 6))
        sns.histplot(self.df[TARGET_COL], kde=True, bins=50)
        plt.title(f"Distribución de '{TARGET_COL}'")
        plt.xlabel("Número de Bicicletas Rentadas")
        plt.ylabel("Frecuencia")
        plt.savefig(self.output_dir / "target_distribution.png")
        plt.close()

    def _plot_demand_by_hour(self):
        """Genera y guarda un gráfico de barras de la demanda promedio por hora."""
        plt.figure(figsize=(12, 6))
        hourly_demand = self.df.groupby("hour")[TARGET_COL].mean()
        sns.barplot(x=hourly_demand.index, y=hourly_demand.values)
        plt.title("Demanda Promedio de Bicicletas por Hora")
        plt.xlabel("Hora del Día")
        plt.ylabel("Promedio de Bicicletas Rentadas")
        plt.savefig(self.output_dir / "demand_by_hour.png")
        plt.close()

    def _plot_correlation_heatmap(self):
        """Genera y guarda un mapa de calor de la correlación entre variables
        numéricas."""
        plt.figure(figsize=(14, 10))

        # Seleccionar solo columnas numéricas para la correlación
        numeric_df = self.df.select_dtypes(include=["number"])

        # Si no hay columnas numéricas, no se puede generar el gráfico
        if numeric_df.empty:
            logger.warning(
                "No se encontraron columnas numéricas para generar el mapa de calor de correlación."
            )
            plt.close()
            return

        corr_matrix = numeric_df.corr()

        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
        plt.title("Mapa de Calor de Correlación de Variables Numéricas")
        plt.savefig(self.output_dir / "correlation_heatmap.png")
        plt.close()


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "seoul_bike_sharing_featured.csv",
    output_dir: Path = FIGURES_DIR,
):
    """Ejecuta el pipeline completo de generación de gráficos de EDA."""
    plotter = PlotGenerator(input_path, output_dir)
    plotter.load_data().generate_plots()


if __name__ == "__main__":
    app()
