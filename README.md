# MLOps


This repository contains experiments and practice exercises related to MLOps concepts and workflows.

The Ml project objective is to predict the bike sharing demand in Seoul based on weather data and holiday information

using a modified version of the following data set https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand

## Project Organization

```
├── Makefile                       <- Convenience commands like `make data` and `make train`
├── README.md                      <- This file; high-level project overview and docs
├── dvc.yaml                       <- DVC pipeline stages and dependencies
├── dvc.lock                       <- Locked DVC artifact versions
├── pyproject.toml                 <- Python project configuration and tool settings
├── poetry.lock                    <- Poetry lockfile for pinned dependencies
├── api/                           <- Minimal API to serve model predictions
│   └── app.py                     <- API entrypoint (Flask/FastAPI app)
├── data/                          <- Data storage for this project
│   ├── external/                  <- Third-party raw data (not generated here)
│   ├── interim/                   <- Intermediate datasets (transformed)
│   ├── processed/                 <- Final datasets used for modeling
│   └── raw/                       <- Original raw data dumps (some tracked by DVC)
|
├── docs/                          <- Project documentation (mkdocs source)
│   └── docs/
│   |   └── docs/
│   └── executive_reports          <- Executive reports created as part of the project phases
├── mlops/                         <- Project source code and utilities
│   ├── config.py                  <- Configuration variables and constants
│   ├── dataset.py                 <- Data loading / download helpers
│   ├── features.py                <- Feature engineering functions
│   ├── plots.py                   <- Plotting utilities for EDA and reports
│   └── modeling/                  <- Training and inference code
│       ├── predict.py             <- Model inference / prediction utilities
│       └── train.py               <- Model training pipeline and scripts
├── models/                        <- Stored trained models and serialized artifacts
├── notebooks/                     <- Jupyter notebooks and exported HTMLs
│   └── HTMLS/                     <- Exported notebook HTML for sharing
├── references/                    <- Reference materials and external docs
├── reports/                       <- Generated reports and analyses
│   └── figures/                   <- Figures used in reports
└── test/                          <- Unit and integration tests
    ├── test_api.py
    ├── test_data.py
    └── test_model.py
```

--------

