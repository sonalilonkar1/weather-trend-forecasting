import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def create_time_series(df, target_col="temperature_celsius", country=None):
    if country is not None and "country" in df.columns:
        df = df[df["country"].str.lower() == country.lower()].copy()
    
    daily = (
        df.groupby(df["last_updated"].dt.date)[target_col]
        .mean()
        .reset_index()
    )

    daily["last_updated"] = pd.to_datetime(daily["last_updated"])
    daily = daily.sort_values("last_updated")

    # Create lag and rolling features
    daily["lag_1"] = daily[target_col].shift(1)
    daily["lag_3"] = daily[target_col].shift(3)
    daily["lag_7"] = daily[target_col].shift(7)
    daily["rolling_3"] = daily[target_col].rolling(3).mean()
    daily["rolling_7"] = daily[target_col].rolling(7).mean()
    daily["day_of_week"] = daily["last_updated"].dt.dayofweek
    daily["month"] = daily["last_updated"].dt.month

    daily = daily.dropna()

    return daily


def evaluate_model(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Avoid division by zero in MAPE
    y_true_safe = np.where(y_true == 0, np.nan, y_true)
    mape = np.nanmean(np.abs((y_true - y_pred) / y_true_safe)) * 100

    r2 = r2_score(y_true, y_pred)

    return {
        "model": name,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2
    }


def train_temperature_forecast_models(df, target_col="temperature_celsius", country=None):
    os.makedirs("outputs/figures", exist_ok=True)

    if target_col not in df.columns:
        raise ValueError(f"{target_col} not found in dataset.")

    daily = create_time_series(df, target_col, country=country)

    features = [
        "lag_1",
        "lag_3",
        "lag_7",
        "rolling_3",
        "rolling_7",
        "day_of_week",
        "month"
    ]

    X = daily[features]
    y = daily[target_col]

    # Time-based split: first 80% train, final 20% test
    split_index = int(len(daily) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    test_dates = daily["last_updated"].iloc[split_index:]

    results = []
    predictions = {}

    # Naive baseline: predict previous day value
    naive_pred = X_test["lag_1"].values
    predictions["Naive Baseline"] = naive_pred
    results.append(evaluate_model("Naive Baseline", y_test, naive_pred))

    # Moving average baseline
    moving_avg_pred = X_test["rolling_7"].values
    predictions["Moving Average"] = moving_avg_pred
    results.append(evaluate_model("Moving Average", y_test, moving_avg_pred))

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    predictions["Linear Regression"] = lr_pred
    results.append(evaluate_model("Linear Regression", y_test, lr_pred))

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    predictions["Random Forest"] = rf_pred
    results.append(evaluate_model("Random Forest", y_test, rf_pred))

    # Gradient Boosting
    gb = GradientBoostingRegressor(random_state=42)
    gb.fit(X_train, y_train)
    gb_pred = gb.predict(X_test)
    predictions["Gradient Boosting"] = gb_pred
    results.append(evaluate_model("Gradient Boosting", y_test, gb_pred))

    # Ensemble
    ensemble_pred = (lr_pred + rf_pred + gb_pred) / 3
    predictions["Ensemble"] = ensemble_pred
    results.append(evaluate_model("Ensemble", y_test, ensemble_pred))

    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv("outputs/model_metrics.csv", index=False)

    # Forecast comparison plot
    plt.figure(figsize=(14, 7))
    plt.plot(test_dates, y_test.values, label="Actual", linewidth=2)

    for name, pred in predictions.items():
        plt.plot(test_dates, pred, label=name, alpha=0.8)

    plt.title("Forecast Model Comparison")
    plt.xlabel("Date")
    plt.ylabel(target_col)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/figures/daily_temperature_forecast_comparison.png")
    plt.close()

    # Feature importance from Random Forest
    feature_importance = pd.DataFrame({
        "feature": features,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False)

    feature_importance.to_csv("outputs/feature_importance.csv", index=False)

    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance["feature"], feature_importance["importance"])
    plt.title("Feature Importance - Random Forest")
    plt.xlabel("Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("outputs/figures/feature_importance.png")
    plt.close()

    return metrics_df