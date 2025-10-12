"""Feature engineering module for the MLOps project."""

import pandas as pd


def create_time_features_from_date(df: pd.DataFrame) -> pd.DataFrame:
    """Creates time-based features from the 'Date' column.

    Args:
        df: The input DataFrame.

    Returns:
        The DataFrame with new time-based features.
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["Weekday"] = df["Date"].dt.day_name()
    return df


def add_is_weekend_and_holiday_or_weekend(df: pd.DataFrame) -> pd.DataFrame:
    """Creates new categorical and binary features is_weekend and is_holiday_or_weekend.

    Args:
        df: The input DataFrame.

    Returns:
        The DataFrame with new categorical and binary features.
    """
    df = df.copy()
    df["is_weekend"] = df["Weekday"].isin(["Saturday", "Sunday"])
    df["is_holiday_or_weekend"] = df["is_weekend"] | (df["Holiday"] == "Holiday")
    return df


def build_features(input_path: str, output_path: str) -> None:
    """Builds features from the raw data and saves the features to the specified output
    path.

    Parameters:
    - input_path: str - Path to the raw data file.
    - output_path: str - Path where the features will be saved.
    """

    # Load clean data
    df = pd.read_csv(input_path)

    # Create features
    df = create_time_features_from_date(df)
    df = add_is_weekend_and_holiday_or_weekend(df)

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
