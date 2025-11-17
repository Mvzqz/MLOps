"""
Tests for the data processing module (mlops/dataset.py).

These tests verify the functionality of the DatasetProcessor class, including
loading, cleaning, preprocessing, and saving data.
"""

import pytest
import pandas as pd
from pathlib import Path
from mlops.dataset import DatasetProcessor
import tempfile
import typer

# Sample data for testing
RAW_DATA = pd.DataFrame({
	'Date': ['01/11/2017', '02/11/2017', '03/11/2017'],
	'Functioning Day': ['Yes', 'No', 'Yes'],
	'Rented Bike Count': [100, 200, 150],
	'hour': [1, None, 3],
	'Extra Col': ['  valor1 ', 'valor2 ', ' valor3']
})

@pytest.fixture
def temp_csv():
	"""Fixture to create a temporary raw CSV file for testing."""
	with tempfile.TemporaryDirectory() as tmpdir:
		file_path = Path(tmpdir) / 'raw.csv'
		RAW_DATA.to_csv(file_path, index=False)
		yield file_path

@pytest.fixture
def output_csv():
	"""Fixture to provide a path for a temporary output CSV file."""
	with tempfile.TemporaryDirectory() as tmpdir:
		file_path = Path(tmpdir) / 'processed.csv'
		yield file_path

def test_load_data_success(temp_csv, output_csv):
	"""Tests successful data loading from a CSV file."""
	processor = DatasetProcessor(temp_csv, output_csv)
	processor.load_data()
	assert isinstance(processor.df, pd.DataFrame)
	assert len(processor.df) == 3

def test_load_data_file_not_found(output_csv):
    """Tests that loading a non-existent file raises a typer.Exit exception."""
    processor = DatasetProcessor(Path('no_existe.csv'), output_csv)
    with pytest.raises(typer.Exit):
        processor.load_data()

def test_clean_data_values(temp_csv, output_csv):
	"""Tests the cleaning of data values, including stripping whitespace and imputing nulls."""
	processor = DatasetProcessor(temp_csv, output_csv)
	processor.load_data()
	processor.clean_data_values()
	# Verify that whitespace was stripped
	assert all(processor.df['Extra Col'].str[0] != ' ')
	assert all(processor.df['Extra Col'].str[-1] != ' ')
	# Verify null imputation
	assert processor.df['hour'].isnull().sum() == 0

def test_normalize_column_names(temp_csv, output_csv):
	"""Tests the normalization of column names to snake_case."""
	processor = DatasetProcessor(temp_csv, output_csv)
	processor.load_data()
	processor._normalize_column_names()
	# Verify snake_case and lowercase
	assert all(col == col.lower() for col in processor.df.columns)
	assert 'functioning_day' in processor.df.columns

def test_preprocess_data(temp_csv, output_csv):
	"""Tests the main data preprocessing steps."""
	processor = DatasetProcessor(temp_csv, output_csv)
	processor.load_data()
	processor.clean_data_values()
	processor.preprocess_data()
	# Verify date conversion
	assert pd.api.types.is_datetime64_any_dtype(processor.df['date'])
	# Verify filtering of non-functioning days
	assert all(processor.df['functioning_day'] == 'Yes')
	# Verify renaming of target column
	assert 'rented_bike_count' in processor.df.columns or 'rented_bike_count' in [col.replace(' ', '_') for col in processor.df.columns]

def test_save_data(temp_csv, output_csv):
	"""Tests that the processed data is saved correctly."""
	processor = DatasetProcessor(temp_csv, output_csv)
	processor.load_data()
	processor.clean_data_values()
	processor.preprocess_data()
	processor.save_data()
	assert output_csv.exists()
	df_saved = pd.read_csv(output_csv)
	assert len(df_saved) == len(processor.df)

def test_clean_data_values_no_df(output_csv):
	"""Tests that calling clean_data_values before loading data raises a ValueError."""
	processor = DatasetProcessor(Path('no_existe.csv'), output_csv)
	with pytest.raises(ValueError):
		processor.clean_data_values()

def test_normalize_column_names_no_df(output_csv):
	"""Tests that calling _normalize_column_names before loading data raises a ValueError."""
	processor = DatasetProcessor(Path('no_existe.csv'), output_csv)
	with pytest.raises(ValueError):
		processor._normalize_column_names()

def test_preprocess_data_no_df(output_csv):
	"""Tests that calling preprocess_data before loading data raises a ValueError."""
	processor = DatasetProcessor(Path('no_existe.csv'), output_csv)
	with pytest.raises(ValueError):
		processor.preprocess_data()

def test_save_data_no_df(output_csv):
	"""Tests that calling save_data before loading data raises a ValueError."""
	processor = DatasetProcessor(Path('no_existe.csv'), output_csv)
	with pytest.raises(ValueError):
		processor.save_data()
