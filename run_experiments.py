import yaml
import subprocess
import json
from pathlib import Path

# Obtiene la ruta raíz del proyecto dinámicamente
PROJECT_ROOT = Path(__file__).resolve().parent

def run_experiment(experiment_name: str, config: dict):
    """
    Ejecuta una corrida de entrenamiento usando subprocess para llamar a train.py.
    """
    print("-" * 50)
    print(f"🚀 Ejecutando experimento: {experiment_name}")
    print("-" * 50)

    train_script_path = PROJECT_ROOT / "mlops" / "modeling" / "train.py"

    # Construye el comando para llamar a train.py
    command = [
        "python",
        str(train_script_path),
    ]

    # Agrega los argumentos desde el archivo de configuración
    for key, value in config.items():
        # Los parámetros de grilla/búsqueda se deben pasar como strings JSON
        if isinstance(value, dict):
            command.extend([f"--{key}", json.dumps(value)])
        else:
            command.extend([f"--{key}", str(value)])

    try:
        # Ejecuta el comando
        subprocess.run(command, check=True, text=True)
        print(f"✅ Experimento '{experiment_name}' completado exitosamente.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error en el experimento '{experiment_name}':")
        print(e)
    except FileNotFoundError:
        print("❌ Error: 'python' no encontrado. Asegúrate de que esté en tu PATH.")

    print("\n")


if __name__ == "__main__":
    experiments_file = PROJECT_ROOT / "experiments.yaml"
    if not experiments_file.exists():
        raise FileNotFoundError(f"El archivo de experimentos no se encontró en {experiments_file}")

    with open(experiments_file, 'r') as f:
        all_experiments = yaml.safe_load(f)

    for name, config in all_experiments.items():
        run_experiment(name, config)

    print("🎉 Todos los experimentos han finalizado.")
