#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = mlops
PYTHON_VERSION = 3.13
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	poetry install
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using flake8, black, and isort (use `make format` to do formatting)
.PHONY: lint
lint:
	poetry run flake8 mlops
	poetry run isort --check --diff mlops
	poetry run black --check mlops

## Format source code with black
.PHONY: format
format:
	poetry run isort mlops
	poetry run black mlops



## Run tests
.PHONY: test
test:
	poetry run pytest tests


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	poetry env use $(PYTHON_VERSION)
	@echo ">>> Poetry environment created. Activate with: "
	@echo '$$(poetry env activate)'
	@echo ">>> Or run commands with:\npoetry run <command>"




#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make dataset
.PHONY: data
data:
	poetry run $(PYTHON_INTERPRETER) mlops/dataset.py

## Make features from cleaned data
.PHONY: features
features: data
	poetry run $(PYTHON_INTERPRETER) mlops/features.py

## Generate plots from featured data
.PHONY: plots
plots: features
	poetry run $(PYTHON_INTERPRETER) mlops/plots.py

## Train model
.PHONY: train
train: features
	poetry run $(PYTHON_INTERPRETER) mlops/modeling/train.py


## Run model inference
.PHONY: predict
predict: train
	poetry run $(PYTHON_INTERPRETER) mlops/modeling/predict.py

## Run the full data science pipeline
.PHONY: run-pipeline
run-pipeline: predict
	@echo "✅ Full pipeline (data -> features -> train -> predict) completed successfully."

## Run the full data science pipeline
.PHONY: experiments
run-pipeline: 
	poetry run $(PYTHON_INTERPRETER) ./run_experiments.py


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@poetry run $(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
