# MLOps Project: Bike Sharing Demand Prediction

Este proyecto implementa un pipeline de Machine Learning de principio a fin para predecir la demanda de bicicletas compartidas en Seúl, utilizando un enfoque robusto de MLOps. El objetivo es demostrar las mejores prácticas para la reproducibilidad, el versionado, el seguimiento de experimentos y la automatización del ciclo de vida de ML.

## 🌟 Características Principales

- **Pipeline Reproducible con DVC:** Todo el flujo de trabajo, desde la limpieza de datos hasta la generación de predicciones, está orquestado con [DVC](https://dvc.org/). Esto garantiza que cada paso sea reproducible con un solo comando.
- **Versionado de Datos y Modelos:** Git se utiliza para el código, mientras que DVC gestiona los datasets y los artefactos de los modelos, manteniendo el repositorio ligero y sincronizado.
- **Seguimiento de Experimentos con MLflow:** Cada experimento de entrenamiento se registra en [MLflow](https://mlflow.org/). Se guardan parámetros, métricas y los propios modelos para facilitar la comparación y el análisis.
- **Promoción Automática del Mejor Modelo:** Un script automatizado consulta los resultados en MLflow, identifica el modelo con el mejor rendimiento (basado en RMSE) y lo "promueve" para su uso en etapas posteriores.
- **Experimentación Basada en Configuración:** El archivo `experiments.yaml` permite definir y lanzar múltiples experimentos (diferentes modelos o hiperparámetros) de forma declarativa y organizada.
- **Código Modular y Estructurado:** El proyecto está organizado en módulos claros para el procesamiento de datos, la ingeniería de características, el entrenamiento y la predicción.

## 🛠️ Herramientas Utilizadas

- **Lenguaje:** Python 3.12+
- **Gestión de Dependencias:** Poetry
- **Pipeline y Versionado de Datos:** DVC (Data Version Control)
- **Seguimiento de Experimentos:** MLflow
- **Frameworks de ML:** Scikit-learn, XGBoost
- **Librerías de Datos:** Pandas, NumPy
- **CLI y Automatización:** Typer, PyYAML

## Modelo usado en la API 

/models/hist_gradient_boosting_regressor/36

## 📂 Estructura del Proyecto

```
├── data/                   # Directorio de datos (rastreado por DVC, no en Git)
│   ├── raw/                # Datos crudos
│   ├── interim/            # Datos intermedios
│   └── processed/          # Datasets listos para el modelado
├── models/                 # Modelos entrenados (rastreado por DVC)
│   └── best_model.pkl      # El mejor modelo promocionado
├── mlops/                  # Código fuente del proyecto
│   ├── dataset.py          # Limpieza y preprocesamiento inicial
│   ├── features.py         # Ingeniería de características
│   ├── modeling/
│   │   ├── train.py        # Script de entrenamiento y tuning
│   │   └── predict.py      # Script para generar predicciones
│   └── config.py           # Configuración centralizada
├── reports/                # Gráficos y reportes generados
├── .dvc/                   # Metadatos de DVC
├── dvc.yaml                # Definición del pipeline de DVC
├── experiments.yaml        # Definición de los experimentos a ejecutar
├── run_experiments.py      # Script para orquestar los experimentos de MLflow
├── run_promote_model.py    # Script para seleccionar y guardar el mejor modelo
├── requirements.txt        # Lista de dependencias
├── pyproject.toml          # Archivo de configuración de Poetry
└── README.md               # Este archivo
```

## 🚀 Cómo Empezar

### 1. Prerrequisitos

- Python 3.12+
- Git
- DVC (`pip install dvc`)
- Poetry (`pip install poetry`)

### 2. Configuración del Entorno

1.  **Clonar el repositorio:**
    ```bash
    git clone <URL_DEL_REPOSITORIO>
    cd <NOMBRE_DEL_REPOSITORIO>
    ```

2.  **Instalar dependencias:**
    Utiliza Poetry para crear un entorno virtual e instalar todas las dependencias.
    ```bash
    poetry install
    ```

3.  **Configurar el almacenamiento remoto de DVC:**
    Este proyecto está configurado para usar un remote de DVC (como DagsHub, S3, Google Drive, etc.). Asegúrate de tener las credenciales configuradas para acceder a él.

4.  **Descargar los datos y modelos:**
    Este comando descargará los datasets y el modelo `best_model.pkl` rastreados por DVC.
    ```bash
    dvc pull
    ```

## ⚙️ Uso del Pipeline

### Ejecutar el Pipeline Completo

Para reproducir todo el pipeline, desde el procesamiento de datos hasta la generación de predicciones y gráficos, ejecuta:

```bash
dvc repro
```

DVC se encargará de ejecutar cada etapa (`dataset`, `features`, `train`, `promote_model`, `predict`, `plots`) en el orden correcto, saltándose las que no hayan cambiado.

### Seguimiento de Experimentos

El pipeline está integrado con MLflow para un seguimiento robusto de los experimentos.

1.  **Definir experimentos:**
    Abre `experiments.yaml` para añadir o modificar experimentos. Puedes definir diferentes modelos o grillas de hiperparámetros.

2.  **Ejecutar el entrenamiento:**
    La etapa `train` del pipeline se encarga de ejecutar todos los experimentos definidos.
    ```bash
    dvc repro train
    ```

3.  **Visualizar los resultados:**
    Inicia la interfaz de usuario de MLflow para comparar las métricas, parámetros y artefactos de cada ejecución.
    ```bash
    mlflow ui
    ```
    Abre tu navegador en `http://127.0.0.1:5000`.

---