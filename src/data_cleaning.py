import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def prepare_weather_dataset(input_path, output_path="outputs/cleaned_weather_data.csv"):
    df = pd.read_csv(input_path)

    original_shape = df.shape

    # Standardize column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    # Convert last_updated to datetime
    if "last_updated" in df.columns:
        df["last_updated"] = pd.to_datetime(df["last_updated"], errors="coerce")
    elif "lastupdated" in df.columns:
        df["last_updated"] = pd.to_datetime(df["lastupdated"], errors="coerce")
    else:
        raise ValueError("Could not find last_updated or lastupdated column.")

    # Remove duplicate rows
    duplicates_before = df.duplicated().sum()
    df = df.drop_duplicates()

    # Separate numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # Handle missing numeric values with median
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Handle missing categorical values
    for col in categorical_cols:
        df[col] = df[col].fillna("Unknown")

    # Drop rows where date is missing
    df = df.dropna(subset=["last_updated"])

    # Cap numeric outliers using 1st and 99th percentiles
    for col in numeric_cols:
        lower = df[col].quantile(0.01)
        upper = df[col].quantile(0.99)
        df[col] = df[col].clip(lower, upper)

    scaled_df = df.copy()

    scalable_cols = [
        col for col in numeric_cols
        if col not in ["latitude", "longitude"]
    ]

    if len(scalable_cols) > 0:
        scaler = StandardScaler()
        scaled_df[scalable_cols] = scaler.fit_transform(scaled_df[scalable_cols])
        scaled_df.to_csv("outputs/normalized_weather_data.csv", index=False)
    
    # Save cleaned data
    df.to_csv(output_path, index=False)

    cleaning_summary = pd.DataFrame({
        "metric": [
            "original_rows",
            "original_columns",
            "cleaned_rows",
            "cleaned_columns",
            "duplicates_removed",
            "normalized_numeric_columns"
        ],
        "value": [
            original_shape[0],
            original_shape[1],
            df.shape[0],
            df.shape[1],
            duplicates_before,
            len(scalable_cols)
        ]
    })

    cleaning_summary.to_csv("outputs/cleaning_summary.csv", index=False)

    return df