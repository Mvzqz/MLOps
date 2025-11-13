import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import pandas as pd
from pathlib import Path
from mlops.dataset import DatasetProcessor
import tempfile
import os
import typer

# Datos de ejemplo para pruebas
RAW_DATA = pd.DataFrame({
	'Date': ['01/11/2017', '02/11/2017', '03/11/2017'],
	'Functioning Day': ['Yes', 'No', 'Yes'],
	'Rented Bike Count': [100, 200, 150],
	'hour': [1, None, 3],
	'Extra Col': ['  valor1 ', 'valor2 ', ' valor3']
})

@pytest.fixture
def temp_csv():
	with tempfile.TemporaryDirectory() as tmpdir:
		file_path = Path(tmpdir) / 'raw.csv'
		RAW_DATA.to_csv(file_path, index=False)
		yield file_path

@pytest.fixture
def output_csv():
	with tempfile.TemporaryDirectory() as tmpdir:
		file_path = Path(tmpdir) / 'processed.csv'
		yield file_path

def test_load_data_success(temp_csv, output_csv):
	processor = DatasetProcessor(temp_csv, output_csv)
	processor.load_data()
	assert isinstance(processor.df, pd.DataFrame)
	assert len(processor.df) == 3

def test_load_data_file_not_found(output_csv):
    processor = DatasetProcessor(Path('no_existe.csv'), output_csv)
    with pytest.raises(typer.Exit):
        processor.load_data()

def test_clean_data_values(temp_csv, output_csv):
	processor = DatasetProcessor(temp_csv, output_csv)
	processor.load_data()
	processor.clean_data_values()
	# Verifica que los espacios fueron eliminados
	assert all(processor.df['Extra Col'].str[0] != ' ')
	assert all(processor.df['Extra Col'].str[-1] != ' ')
	# Verifica imputación de nulos
	assert processor.df['hour'].isnull().sum() == 0

def test_normalize_column_names(temp_csv, output_csv):
	processor = DatasetProcessor(temp_csv, output_csv)
	processor.load_data()
	processor._normalize_column_names()
	# Verifica snake_case y minúsculas
	assert all(col == col.lower() for col in processor.df.columns)
	assert 'functioning_day' in processor.df.columns

def test_preprocess_data(temp_csv, output_csv):
	processor = DatasetProcessor(temp_csv, output_csv)
	processor.load_data()
	processor.clean_data_values()
	processor.preprocess_data()
	# Verifica conversión de fecha
	assert pd.api.types.is_datetime64_any_dtype(processor.df['date'])
	# Verifica filtrado de días no funcionales
	assert all(processor.df['functioning_day'] == 'Yes')
	# Verifica renombrado de columna objetivo
	assert 'rented_bike_count' in processor.df.columns or 'rented_bike_count' in [col.replace(' ', '_') for col in processor.df.columns]

def test_save_data(temp_csv, output_csv):
	processor = DatasetProcessor(temp_csv, output_csv)
	processor.load_data()
	processor.clean_data_values()
	processor.preprocess_data()
	processor.save_data()
	assert output_csv.exists()
	df_saved = pd.read_csv(output_csv)
	assert len(df_saved) == len(processor.df)

def test_clean_data_values_no_df(output_csv):
	processor = DatasetProcessor(Path('no_existe.csv'), output_csv)
	with pytest.raises(ValueError):
		processor.clean_data_values()

def test_normalize_column_names_no_df(output_csv):
	processor = DatasetProcessor(Path('no_existe.csv'), output_csv)
	with pytest.raises(ValueError):
		processor._normalize_column_names()

def test_preprocess_data_no_df(output_csv):
	processor = DatasetProcessor(Path('no_existe.csv'), output_csv)
	with pytest.raises(ValueError):
		processor.preprocess_data()

def test_save_data_no_df(output_csv):
	processor = DatasetProcessor(Path('no_existe.csv'), output_csv)
	with pytest.raises(ValueError):
		processor.save_data()
