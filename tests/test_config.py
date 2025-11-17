"""
Tests for the project's configuration module (mlops/config.py).

Verifies that the configuration loads correctly and contains essential
paths and parameters.
"""

from mlops import config

def test_config_paths_exist():
    """Tests that essential configuration attributes exist and are correctly typed."""
    assert hasattr(config, "RAW_DATA_DIR")
    assert hasattr(config, "MODELS_DIR")
    assert isinstance(config.PARAM_GRIDS, dict)
    assert "hist_gradient_boosting_regressor" in config.PARAM_GRIDS
    assert config.DEFAULT_MODEL_NAME in config.PARAM_GRIDS