import yaml
import subprocess
import sys
import json
from pathlib import Path

# Get the project root path dynamically
PROJECT_ROOT = Path(__file__).resolve().parent

def run_experiment(experiment_name: str, config: dict):
    """
    Runs a training run using subprocess to call train.py.
    """
    print("-" * 50)
    print(f"Running experimet: {experiment_name}")
    print("-" * 50)

    train_script_path = PROJECT_ROOT / "mlops" / "modeling" / "train.py"

    # Build the command to call train.py
    command = [
        sys.executable,
        str(train_script_path),
        "--run-name", experiment_name,  # Pass the experiment name as the run name
    ]

    # Add arguments from the configuration file
    for key, value in config.items():
        arg_name = f"--{key.replace('_', '-')}"
        # Grid/search parameters must be passed as JSON strings
        if isinstance(value, dict):
            command.extend([arg_name, json.dumps(value)])
        else:
            command.extend([arg_name, str(value)])

    try:
        # Run the command
        subprocess.run(command, check=True, text=True)
        print(f"Experiment '{experiment_name}' completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error in experiment '{experiment_name}':")
        print(e)
    except FileNotFoundError:
        print("Error: 'python' not found. Make sure it is in your PATH.")

    print("\n")


if __name__ == "__main__":
    experiments_file = PROJECT_ROOT / "experiments.yaml"
    if not experiments_file.exists():
        raise FileNotFoundError(f"Experiments file not found at {experiments_file}")

    with open(experiments_file, 'r') as f:
        all_experiments = yaml.safe_load(f)

    for name, config in all_experiments.items():
        run_experiment(name, config)

    print("All experiments have finished.")
