#Verifica la correcta generación de gráficos sin errores.

import pytest
import pandas as pd
from mlops.plots import PlotGenerator
import matplotlib.pyplot as plt

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'hour': [0, 1, 2, 3],
        'rented_bike_count': [100, 150, 200, 250],
        'temperature': [5, 7, 9, 12]
    })

def test_generate_heatmap(sample_data):
    pg = PlotGenerator(sample_data)
    fig = pg.generate_heatmap()
    assert isinstance(fig, plt.Figure)

def test_plot_demand_by_hour(sample_data):
    pg = PlotGenerator(sample_data)
    fig = pg.plot_demand_by_hour()
    assert isinstance(fig, plt.Figure)