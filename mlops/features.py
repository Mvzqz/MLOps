"""
Módulo para ingeniería de características del dataset Seoul Bike Sharing Demand.
Incluye creación de nuevas variables, codificación categórica y escalado.
"""

from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import typer
from loguru import logger
from tqdm import tqdm
from mlops.config import PROCESSED_DATA_DIR, INTERIM_DATA_DIR

# Inicialización de Typer y logging
app = typer.Typer()


def crear_pipeline_preprocesamiento(vars_numericas, vars_categoricas):
    """
    Crea un pipeline de preprocesamiento para variables numéricas y categóricas.
    """
    logger.info("⚙️ Creando pipeline de preprocesamiento...")

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("numerico", num_pipeline, vars_numericas),
        ("categorico", cat_pipeline, vars_categoricas)
    ])

    logger.info("✅ Pipeline de preprocesamiento creado con éxito.")
    return preprocessor


def agregar_caracteristicas_personalizadas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega variables derivadas específicas para el dataset Seoul Bike Sharing Demand.
    """
    logger.info("🧠 Creando nuevas características derivadas...")

    # Interacción temperatura-humedad
    df["Temp_Humidity_Interaction"] = df["Temperature(°C)"] * df["Humidity(%)"]

    # Codificación de hora pico
    df["Is_Peak_Hour"] = df["Hour"].apply(lambda x: 1 if (7 <= x <= 9) or (17 <= x <= 20) else 0)

    # Codificación de estación del año
    le = LabelEncoder()
    df["Season_Encoded"] = le.fit_transform(df["Seasons"])

    # Indicador de fin de semana
    if "Functioning Day" in df.columns:
        df["Weekend_Flag"] = df["Functioning Day"].apply(lambda x: 0 if x.strip().lower() == "yes" else 1)

    logger.info("✅ Nuevas características generadas correctamente.")
    return df


def transformar_datos(preprocessor, X):
    """
    Aplica el pipeline de preprocesamiento a un DataFrame de entrada
    y devuelve el resultado con nombres de columnas.
    """
    logger.info("🔄 Transformando datos con el pipeline...")
    X_trans = preprocessor.fit_transform(X)

    # Obtener nombres de columnas del transformador
    feature_names = []
    try:
        feature_names = preprocessor.get_feature_names_out()
    except AttributeError:
        # Compatibilidad con versiones anteriores de scikit-learn
        for name, trans, cols in preprocessor.transformers_:
            if hasattr(trans, "get_feature_names_out"):
                fn = trans.get_feature_names_out(cols)
            else:
                fn = cols
            feature_names.extend(fn)

    X_trans_df = pd.DataFrame(X_trans, columns=feature_names)
    logger.info(f"✅ Transformación completada. Nueva forma: {X_trans_df.shape}")
    return X_trans_df


@app.command()
def main(
    input_path: Path = INTERIM_DATA_DIR / "processed_data.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv"
):
    """
    Genera características transformadas a partir del dataset procesado,
    aplicando un pipeline automático de preprocesamiento.
    """
    logger.info(f"📂 Leyendo datos desde: {input_path}")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        logger.error(f"No se encontró el archivo en {input_path}")
        raise typer.Exit(code=1)

    logger.info("🔍 Explorando columnas del dataset...")
    logger.info(f"Columnas disponibles: {list(df.columns)}")

    # Agregar nuevas características derivadas
    df = agregar_caracteristicas_personalizadas(df)

    print(df.columns)

    # Identificar columnas numéricas y categóricas
    vars_numericas = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    vars_categoricas = df.select_dtypes(include=["object", "category"]).columns.tolist()

    logger.info(f"Variables numéricas detectadas: {vars_numericas}")
    logger.info(f"Variables categóricas detectadas: {vars_categoricas}")

    # Crear pipeline y transformar datos
    preprocessor = crear_pipeline_preprocesamiento(vars_numericas, vars_categoricas)

    logger.info("🚀 Iniciando generación de características...")
    with tqdm(total=100, desc="Procesando pipeline") as pbar:
        features = transformar_datos(preprocessor, df)
        pbar.update(100)

    # Guardar con nombres de columnas
    features.to_csv(output_path, index=False)

    logger.info(f"💾 Archivo con características guardado en: {output_path}")
    logger.info("🎯 Proceso completado exitosamente.")


if __name__ == "__main__":
    app()
