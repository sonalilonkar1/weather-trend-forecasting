import argparse
import os

from src.data_cleaning import prepare_weather_dataset
from src.eda import create_weather_exploration_charts
from src.forecasting import train_temperature_forecast_models
from src.advanced_analysis import create_advanced_weather_summaries

def main():
    parser = argparse.ArgumentParser(description="Weather Trend Forecasting Project")
    parser.add_argument(
        "--data",
        type=str,
        default="data/GlobalWeatherRepository.csv",
        help="Path to the weather CSV file"
    )

    args = parser.parse_args()

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)

    print("Step 1: Loading and preparing the Global Weather Repository dataset...")
    df = prepare_weather_dataset(args.data)

    print("Step 2: Creating temperature, precipitation, and correlation charts...")
    create_weather_exploration_charts(df)

    print("Step 3: Training daily temperature forecasting models...")
    metrics = train_temperature_forecast_models(df, target_col="temperature_celsius", country=None)

    print("Step 4: Running advanced analysis for anomaly, climate, air-quality, and geography summaries...")
    create_advanced_weather_summaries(df)

    print("\nProject completed successfully.")
    print("\nModel metrics:")
    print(metrics)

    print("\nCheck outputs/ and outputs/figures/ for results.")


if __name__ == "__main__":
    main()