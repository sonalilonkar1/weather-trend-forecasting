import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.ensemble import IsolationForest


def anomaly_detection(df):
    os.makedirs("outputs/figures", exist_ok=True)

    possible_cols = [
        "temperature_celsius",
        "precip_mm",
        "humidity",
        "wind_kph",
        "pressure_mb"
    ]

    anomaly_cols = [col for col in possible_cols if col in df.columns]

    if len(anomaly_cols) < 2:
        return None

    data = df[anomaly_cols].copy()

    model = IsolationForest(contamination=0.03, random_state=42)
    df["anomaly"] = model.fit_predict(data)

    anomalies = df[df["anomaly"] == -1]
    anomalies.to_csv("outputs/anomaly_summary.csv", index=False)

    if "temperature_celsius" in df.columns:
        iqr_temperature_outliers = detect_iqr_outliers(df, "temperature_celsius")
        iqr_temperature_outliers.to_csv(
            "outputs/iqr_temperature_outliers.csv",
            index=False
        )
        plt.figure(figsize=(12, 6))
        plt.scatter(df["last_updated"], df["temperature_celsius"], s=8, label="Normal")
        plt.scatter(
            anomalies["last_updated"],
            anomalies["temperature_celsius"],
            s=12,
            label="Anomaly"
        )
        plt.title("Temperature Anomaly Detection")
        plt.xlabel("Date")
        plt.ylabel("Temperature Celsius")
        plt.legend()
        plt.tight_layout()
        plt.savefig("outputs/figures/anomaly_detection.png")
        plt.close()

    return anomalies


def climate_analysis(df):
    df["month"] = df["last_updated"].dt.month

    if "temperature_celsius" not in df.columns:
        return None

    group_cols = ["month"]

    if "country" in df.columns:
        group_cols = ["country", "month"]

    value_cols = []
    for col in ["temperature_celsius", "precip_mm", "humidity", "wind_kph"]:
        if col in df.columns:
            value_cols.append(col)

    climate_summary = (
        df.groupby(group_cols)[value_cols]
        .mean()
        .reset_index()
    )

    climate_summary.to_csv("outputs/climate_summary.csv", index=False)

    monthly = df.groupby("month")[value_cols].mean().reset_index()

    plt.figure(figsize=(10, 6))
    plt.plot(monthly["month"], monthly["temperature_celsius"], marker="o")
    plt.title("Average Monthly Temperature Pattern")
    plt.xlabel("Month")
    plt.ylabel("Temperature Celsius")
    plt.tight_layout()
    plt.savefig("outputs/figures/climate_monthly_patterns.png")
    plt.close()

    return climate_summary


def air_quality_analysis(df):
    air_quality_cols = [
        col for col in df.columns
        if "air_quality" in col.lower() or "pm2" in col.lower() or "pm10" in col.lower()
    ]

    weather_cols = [
        col for col in [
            "temperature_celsius",
            "humidity",
            "wind_kph",
            "pressure_mb",
            "precip_mm"
        ]
        if col in df.columns
    ]

    if len(air_quality_cols) == 0 or len(weather_cols) == 0:
        return None

    selected_cols = weather_cols + air_quality_cols
    corr = df[selected_cols].corr()

    air_weather_corr = corr.loc[weather_cols, air_quality_cols]
    air_weather_corr.to_csv("outputs/air_quality_weather_correlations.csv")

    plt.figure(figsize=(12, 8))
    sns.heatmap(air_weather_corr, annot=False, cmap="coolwarm", center=0)
    plt.title("Air Quality vs Weather Correlations")
    plt.tight_layout()
    plt.savefig("outputs/figures/air_quality_correlation_heatmap.png")
    plt.close()

    return air_weather_corr


def geographical_analysis(df):
    geo_cols = []

    for col in ["country", "location_name", "latitude", "longitude", "temperature_celsius", "precip_mm"]:
        if col in df.columns:
            geo_cols.append(col)

    if "country" not in df.columns:
        return None

    value_cols = []
    for col in ["temperature_celsius", "precip_mm", "humidity", "wind_kph"]:
        if col in df.columns:
            value_cols.append(col)

    geographical_summary = (
        df.groupby("country")[value_cols]
        .mean()
        .reset_index()
        .sort_values(value_cols[0], ascending=False)
    )

    geographical_summary.to_csv("outputs/geographical_summary.csv", index=False)

    if "temperature_celsius" in df.columns:
        top_temp = geographical_summary.head(15)

        plt.figure(figsize=(12, 6))
        plt.bar(top_temp["country"], top_temp["temperature_celsius"])
        plt.title("Top 15 Countries by Average Temperature")
        plt.xlabel("Country")
        plt.ylabel("Temperature Celsius")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("outputs/figures/geographical_temperature_patterns.png")
        plt.close()

    return geographical_summary


def create_advanced_weather_summaries(df):
    anomalies = anomaly_detection(df)
    climate = climate_analysis(df)
    air_quality = air_quality_analysis(df)
    geography = geographical_analysis(df)

    return {
        "anomalies": anomalies,
        "climate": climate,
        "air_quality": air_quality,
        "geography": geography
    }
    
def detect_iqr_outliers(df, column):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]