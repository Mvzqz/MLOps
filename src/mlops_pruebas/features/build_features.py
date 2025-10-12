"""Feature engineering module for the MLOps project."""

import pandas as pd


def build_features(input_path: str, output_path: str) -> None:
    """Builds features from the raw data and saves the features to the specified output
    path.

    Parameters:
    - input_path: str - Path to the raw data file.
    - output_path: str - Path where the features will be saved.
    """

    # Load clean data
    df = pd.read_csv(input_path)

    # Feature engineering steps
    # Save features
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python build_features.py <input_path> <output_path>")
    else:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        build_features(input_path, output_path)
