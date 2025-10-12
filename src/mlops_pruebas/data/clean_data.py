"""Data cleaning module for the MLOps project."""

import pandas as pd


def clean_data(input_path: str, output_path: str) -> None:
    """Cleans the raw data and saves the cleaned data to the specified output path.

    Parameters:
    - input_path: str - Path to the raw data file.
    - output_path: str - Path where the cleaned data will be saved.
    """

    # Load raw data
    df = pd.read_csv(input_path)

    # Save cleaned data
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python clean_data.py <input_path> <output_path>")
    else:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        clean_data(input_path, output_path)
