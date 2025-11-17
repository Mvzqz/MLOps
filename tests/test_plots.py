"""
Tests for the plot generation module (mlops/plots.py).

Verifies that EDA plots are generated correctly without errors.
"""

import pytest
import pandas as pd
from pathlib import Path
from mlops.plots import PlotGenerator
import matplotlib.pyplot as plt

@pytest.fixture
def sample_data():
    """Provides a sample DataFrame for plotting tests."""
    return pd.DataFrame({
        'hour': [0, 1, 2, 3],
        'rented_bike_count': [100, 150, 200, 250],
        'temperature': [5, 7, 9, 12]
    })

def _make_plot_generator(tmp_path: Path, df: pd.DataFrame) -> PlotGenerator:
    """Helper to create a PlotGenerator instance with temporary paths."""
    input_path = tmp_path / "data.csv"
    df.to_csv(input_path, index=False)
    output_dir = tmp_path / "figures"
    pg = PlotGenerator(input_path=input_path, output_dir=output_dir)
    pg.load_data()
    return pg


def test_generate_all_plots(sample_data, tmp_path):
    """Tests that all defined plots are generated and saved successfully."""
    pg = _make_plot_generator(tmp_path, sample_data)
    pg.generate_plots()
    for expected_file in [
        "target_distribution.png",
        "demand_by_hour.png",
        "correlation_heatmap.png",
    ]:
        assert (pg.output_dir / expected_file).exists(), f"{expected_file} was not generated"
