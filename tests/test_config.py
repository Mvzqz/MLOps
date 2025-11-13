#Verifica que la configuración cargue correctamente rutas y parámetros.
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mlops import config

def test_config_paths_exist():
    assert hasattr(config, "RAW_DATA_DIR")
    assert hasattr(config, "MODELS_DIR")
    assert isinstance(config.PARAM_GRIDS, dict)
    assert "hist_gradient_boosting_regressor" in config.PARAM_GRIDS
    assert config.DEFAULT_MODEL_NAME in config.PARAM_GRIDS