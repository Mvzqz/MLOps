<<<<<<< HEAD
"""Data cleaning module for the MLOps project."""

import numpy as np
import pandas as pd


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the dataframe with normalized column names."""
    df = df.copy()
    df.rename(
        columns={
            c: c.strip().replace("\xa0", " ").replace("  ", " ").strip()
            for c in df.columns
        },
        inplace=True,
    )
    return df


def parse_date_hour(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the original date and hour columns in-place:
    - Converts the date column to datetime
    - Converts the hour column to numeric
    """
    df = df.copy()

    # Find date column
    date_cols = [c for c in df.columns if "date" in c.lower()]
    if date_cols:
        df[date_cols[0]] = pd.to_datetime(
            df[date_cols[0]], errors="coerce", dayfirst=True
        )

    # Find hour column
    hour_cols = [c for c in df.columns if "hour" in c.lower()]
    if hour_cols:
        df[hour_cols[0]] = pd.to_numeric(df[hour_cols[0]], errors="coerce")

    return df


def guess_target(cols: list) -> str | None:
    """Return the first column name containing 'rented' and 'count', or None."""
    for c in cols:
        if "rented" in c.lower() and "count" in c.lower():
            return c
    return None


def clean_df(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Clean dataframe: strip strings, convert numeric-like objects, clip outliers."""
    df = df.copy()

    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype(str).str.strip().replace({"nan": np.nan})

    for c in df.columns:
        if c != target_col and df[c].dtype == object:
            num = pd.to_numeric(
                df[c].str.replace(",", "").str.replace("%", ""), errors="coerce"
            )
            if num.notna().sum() >= 0.5 * len(df):
                df[c] = num

    for c in df.select_dtypes(include=[np.number]).columns:
        q1, q99 = df[c].quantile(0.01), df[c].quantile(0.99)
        if pd.notna(q1) and pd.notna(q99) and q99 > q1:
            df[c] = df[c].clip(q1, q99)

    return df


def detect_categoricals(df: pd.DataFrame, target_col: str) -> list:
    """Return sorted list of categorical columns, excluding target and '__' columns."""
    cats = list(df.select_dtypes(include=["object", "category", "bool"]).columns)

    for c in df.select_dtypes(include=[np.number]).columns:
        if df[c].nunique(dropna=True) <= 20 and c != target_col:
            cats.append(c)

    return sorted([c for c in set(cats) if not c.startswith("__")])


def clean_data(input_path: str, output_path: str) -> None:
    """Cleans raw data and saves the cleaned data to the specified output path.

    Parameters:
    - input_path: str - Path to the raw data file.
    - output_path: str - Path where the cleaned data will be saved.
    """

    df = pd.read_csv(input_path)
    df = normalize_cols(df)
    df = parse_date_hour(df)
    target_col = guess_target(df.columns.tolist())
    df = clean_df(df, target_col)
    categorical_cols = detect_categoricals(df, target_col)
    for col in categorical_cols:
        df[col] = df[col].astype("category")
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python clean_data.py <input_path> <output_path>")
    else:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        clean_data(input_path, output_path)
=======
from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from mlops.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Processing dataset complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
>>>>>>> origin/main
