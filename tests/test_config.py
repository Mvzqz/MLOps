#Verifica que la configuración cargue correctamente rutas y parámetros.

from mlops.config import Config

def test_config_paths_exist():
    cfg = Config()
    assert hasattr(cfg, 'RAW_DATA_PATH')
    assert hasattr(cfg, 'MODEL_DIR')
    assert isinstance(cfg.to_dict(), dict)