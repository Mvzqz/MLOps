# MLOps_pruebas

This repository contains experiments and practice exercises related to MLOps concepts and workflows.

The Ml project objective is to predict the bike sharing demand in Seoul based on weather data and holiday information

using a modified version of the following data set https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand

## Project Structure

```
MLOps_pruebas/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── exploration/
│   └── experiments/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── preprocessing.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── train.py
│   ├── pipelines/
│   │   ├── __init__.py
│   │   └── pipeline.py
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
├── tests/
│   ├── test_data.py
│   ├── test_models.py
│   └── test_pipelines.py
├── .github/
│   └── workflows/
│       └── ci.yml
├── pyproject.toml
└── README.md
```

## Features

- Experimentation with MLOps tools and best practices
- Example pipelines for model training and deployment
- Automated testing and CI/CD integration

## Getting Started

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/MLOps_pruebas.git
    cd MLOps_pruebas
    ```
2. Install dependencies using Poetry:
    ```bash
    poetry install
    ```
