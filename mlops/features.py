"""Módulo para ingeniería de características y preprocesamiento del dataset Seoul Bike
Sharing.

Integra los pipelines numéricos y categóricos definidos originalmente,
junto con generación de nuevas variables y registro de logs para
trazabilidad.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from tqdm import tqdm
import typer

from mlops.config import PROCESSED_DATA_DIR

# Inicializar aplicación y logger


app = typer.Typer()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(name)

# ----------------------------------------------------------
# Creación de Pipelines de Preprocesamiento
# ---------------------------------------------------------


def crear_pipeline_preprocesamiento(vars_numericas, vars_categoricas):
    """Crea un pipeline de preprocesamiento para variables numéricas y categóricas.

    Args:
        vars_numericas (list): Columnas numéricas.
        vars_categoricas (list): Columnas categóricas.

    Returns:
        ColumnTransformer: Pipeline completo de preprocesamiento.
    """
    logger.info("⚙️ Creando pipeline de preprocesamiento...")

    # Pipeline numérico
    num_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    # Pipeline categórico
    cat_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Composición general
    preprocessor = ColumnTransformer(
        [
            ("numerico", num_pipeline, vars_numericas),
            ("categorico", cat_pipeline, vars_categoricas),
        ]
    )

    logger.info("✅ Pipeline de preprocesamiento creado correctamente.")
    return preprocessor


# ----------------------------------------------------------
# Generación de nuevas características personalizadas
# ----------------------------------------------------------


def agregar_caracteristicas_personalizadas(df: pd.DataFrame) -> pd.DataFrame:
    """Crea nuevas columnas derivadas del dataset base.

    Args:
        df (pd.DataFrame): Dataset original.

    Returns:
        pd.DataFrame: Dataset con nuevas columnas agregadas.
    """
    logger.info("🧩 Agregando características derivadas...")

    if "Temperature(°C)" in df.columns and "Humidity(%)" in df.columns:
        df["Temp_Humidity_Interaction"] = df["Temperature(°C)"] * df["Humidity(%)"]

    if "Hour" in df.columns:
        df["Is_Peak_Hour"] = df["Hour"].apply(
            lambda x: 1 if (7 <= x <= 9) or (17 <= x <= 20) else 0
        )

    if "Seasons" in df.columns:
        le = LabelEncoder()
        df["Season_Encoded"] = le.fit_transform(df["Seasons"])

    if "Functioning Day" in df.columns:
        df["Weekend_Flag"] = df["Functioning Day"].apply(
            lambda x: 0 if x.strip().lower() == "yes" else 1
        )

    logger.info("✅ Nuevas características generadas correctamente.")
    return df


# ----------------------------------------------------------
# Aplicar pipeline de transformación
# ----------------------------------------------------------


def transformar_datos(preprocessor, X: pd.DataFrame) -> np.ndarray:
    """Aplica el pipeline de preprocesamiento a los datos de entrada.

    Args:
        preprocessor: Pipeline ajustado.
        X (pd.DataFrame): Datos originales.

    Returns:
        np.ndarray: Datos transformados listos para modelado.
    """
    logger.info("🔄 Aplicando transformación con el pipeline...")
    X_trans = preprocessor.fit_transform(X)
    logger.info(f"✅ Transformación completada. Nueva forma: {X_trans.shape}")
    return X_trans


# ----------------------------------------------------------
# Ejecución principal del script con Typer
# ----------------------------------------------------------


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "processed_data.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
):
    """Ejecuta el flujo completo de ingeniería de características:

    - Lectura del dataset procesado
    - Generación de nuevas características
    - Creación y aplicación de pipelines
    - Exportación de dataset transformado
    """

    logger.info(f"📂 Leyendo datos desde: {input_path}")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        logger.error(f"❌ No se encontró el archivo en {input_path}")
        raise typer.Exit(code=1)

    logger.info("🔍 Explorando columnas disponibles...")
    logger.info(f"Columnas detectadas: {list(df.columns)}")

    # Crear nuevas variables
    df = agregar_caracteristicas_personalizadas(df)

    # Identificar variables por tipo
    vars_numericas = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    vars_categoricas = df.select_dtypes(include=["object", "category"]).columns.tolist()

    logger.info(f"📊 Variables numéricas: {vars_numericas}")
    logger.info(f"🔠 Variables categóricas: {vars_categoricas}")

    # Crear y aplicar pipeline
    preprocessor = crear_pipeline_preprocesamiento(vars_numericas, vars_categoricas)

    with tqdm(total=100, desc="Procesando pipeline") as pbar:
        X_trans = transformar_datos(preprocessor, df)
        pbar.update(100)

    # Convertir a DataFrame y guardar
    features = pd.DataFrame(X_trans)
    features.to_csv(output_path, index=False)

    logger.info(f"💾 Archivo de características guardado en: {output_path}")
    logger.info("🎯 Flujo de ingeniería de características completado exitosamente.")


if __name__ == "main":
    app()
