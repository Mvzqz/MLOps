"""
This module is responsible for creating new features from the cleaned dataset.
It includes time-based features, cyclical features, and lag/rolling features
to capture temporal patterns in the data.
"""
import numpy as np
import pandas as pd
import typer
from loguru import logger

from mlops.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers features from the input DataFrame.

    This function adds:
    1. Time-based features (year, month, day, dayofweek, is_weekend).
    2. Cyclical features for 'hour' and 'month' to capture cyclical patterns.
    3. Interaction features like 'is_rush_hour' and 'is_holiday_or_weekend'.
    4. Lag features for the target variable to capture recent trends.
    5. Rolling window features to capture moving averages.

    Args:
        df (pd.DataFrame): The input DataFrame with a 'date' column.

    Returns:
        pd.DataFrame: The DataFrame with new engineered features.
    """
    df = df.copy()

    # Ensure 'date' is a datetime object
    df["date"] = pd.to_datetime(df["date"])
    
    # Drop rows with invalid dates as they are not useful
    if df["date"].isnull().any():
        logger.warning(f"Dropping {df['date'].isnull().sum()} rows with invalid dates.")
        df.dropna(subset=["date"], inplace=True)

    # 1. Time-based features
    df["year"] = df["date"].dt.year.astype("Int64")
    df["month"] = df["date"].dt.month.astype("Int64")
    df["dayofyear"] = df["date"].dt.dayofyear.astype("Int64")
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype("Int64")
    df["quarter"] = df["date"].dt.quarter.astype("Int64")
    df["day"] = df["date"].dt.day.astype("Int64")
    df["dayofweek"] = df["date"].dt.dayofweek.astype("Int64")  # Monday=0, Sunday=6
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype("Int64")

    # Ensure 'hour' is a numeric type and handle potential errors
    if "hour" in df.columns:
        df["hour"] = pd.to_numeric(df["hour"], errors="coerce")
        # Clip values to be within the valid range [0, 23] and fill NaNs
        df["hour"] = df["hour"].clip(0, 23).fillna(df["hour"].median())
        df["hour"] = df["hour"].astype("Int64")

    # 2. Cyclical features
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12.0)
    df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0)

    # 3. Interaction & derived features
    df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype("Int64")

    # Handle 'holiday' column if it exists
    if "holiday" in df.columns:
        is_holiday = df["holiday"].astype(str).str.lower().isin(["holiday", "yes", "1", "true"])
        df["is_holiday_or_weekend"] = (df["is_weekend"] == 1) | is_holiday
        df["is_holiday_or_weekend"] = df["is_holiday_or_weekend"].astype("Int64")
        df = df.drop(columns=["holiday"], errors="ignore")

    # Ensure weather columns are numeric before creating interactions
    weather_cols = [
        "temperature_c",
        "humidity",
        "wind_speed_m_s",
        "visibility_10m",
        "dew_point_temperature_c",
        "solar_radiation_mj_m2",
    ]
    for col in weather_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Interaction and polynomial features from the notebook
    if "temperature_c" in df.columns and "humidity" in df.columns:
        df["temp_humidity"] = df["temperature_c"] * df["humidity"]
        df["comfort_index"] = df["temperature_c"] - df["humidity"] / 5.0
    if "wind_speed_m_s" in df.columns and "humidity" in df.columns:
        df["wind_discomfort"] = df["wind_speed_m_s"] * df["humidity"]
    if "temperature_c" in df.columns:
        df["temp_sq"] = df["temperature_c"] ** 2

    # Drop the original date column as it's no longer needed for modeling
    df = df.drop(columns=["date"], errors="ignore")

    # 4. Lag and Rolling Window Features (requires sorting by time)
    # This assumes the data is already sorted by date and hour from the cleaning step.
    target = "rented_bike_count"
    # if target in df.columns:
    #     # Ensure the target column is numeric
    if target in df.columns:
        df[target] = pd.to_numeric(df[target], errors="coerce")
        #Windzorize the target to handle potential outliers
        q_low = df[target].quantile(0.01)
        q_high = df[target].quantile(0.99)
        df[target] = df[target].clip(lower=q_low, upper=q_high)
    df.reset_index(drop=True, inplace=True)

    return df


@app.command()
def main():
    """
    Main function to run the feature engineering process.
    Loads the interim data, creates features, and saves the processed data.
    """
    logger.info("Starting feature engineering...")
    input_path = INTERIM_DATA_DIR / "seoul_bike_sharing_cleaned.csv"
    output_path = PROCESSED_DATA_DIR / "seoul_bike_sharing_featured.csv"

    df_interim = pd.read_csv(input_path)
    df_featured = create_features(df_interim)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_featured.to_csv(output_path, index=False)
    logger.success(f"Feature engineering complete. Data saved to {output_path}")


if __name__ == "__main__":
    app()