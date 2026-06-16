import matplotlib.pyplot as plt
import seaborn as sns
import os


def create_weather_exploration_charts(df):
    os.makedirs("outputs/figures", exist_ok=True)

    # Dataset summary
    df.describe().to_csv("outputs/dataset_summary.csv")

    # Missing values chart
    missing = df.isnull().mean().sort_values(ascending=False) * 100
    missing = missing[missing > 0]

    if len(missing) > 0:
        plt.figure(figsize=(10, 6))
        missing.head(20).plot(kind="bar")
        plt.title("Top Missing Value Percentages")
        plt.ylabel("Missing %")
        plt.tight_layout()
        plt.savefig("outputs/figures/missing_values.png")
        plt.close()

    # Temperature trend
    if "temperature_celsius" in df.columns:
        daily_temp = (
            df.groupby(df["last_updated"].dt.date)["temperature_celsius"]
            .mean()
            .reset_index()
        )

        plt.figure(figsize=(12, 6))
        plt.plot(daily_temp["last_updated"], daily_temp["temperature_celsius"])
        plt.title("Average Temperature Trend Over Time")
        plt.xlabel("Date")
        plt.ylabel("Temperature Celsius")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("outputs/figures/temperature_trend.png")
        plt.close()

    # Precipitation trend
    if "precip_mm" in df.columns:
        daily_precip = (
            df.groupby(df["last_updated"].dt.date)["precip_mm"]
            .mean()
            .reset_index()
        )

        plt.figure(figsize=(12, 6))
        plt.plot(daily_precip["last_updated"], daily_precip["precip_mm"])
        plt.title("Average Precipitation Trend Over Time")
        plt.xlabel("Date")
        plt.ylabel("Precipitation mm")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("outputs/figures/precipitation_trend.png")
        plt.close()

    # Correlation heatmap
    numeric_df = df.select_dtypes(include="number")

    if numeric_df.shape[1] > 1:
        corr = numeric_df.corr()

        plt.figure(figsize=(14, 10))
        sns.heatmap(corr, cmap="coolwarm", center=0)
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig("outputs/figures/correlation_heatmap.png")
        plt.close()

    # Top countries
    if "country" in df.columns:
        country_counts = df["country"].value_counts().head(15)

        plt.figure(figsize=(12, 6))
        country_counts.plot(kind="bar")
        plt.title("Top 15 Countries by Record Count")
        plt.xlabel("Country")
        plt.ylabel("Record Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("outputs/figures/top_countries.png")
        plt.close()