import streamlit as st
import pandas as pd
import os
from PIL import Image


st.set_page_config(
    page_title="Weather Trend Forecasting",
    layout="wide"
)

st.title("Weather Trend Forecasting Dashboard")

st.markdown("""
## PM Accelerator Mission

By making industry-leading tools and education available to individuals from all backgrounds, we level the playing field for future PM leaders. 
This is the PM Accelerator motto, as we grant aspiring and experienced PMs what they need most – Access. 
We introduce you to industry leaders, surround you with the right PM ecosystem, and discover the new world of AI product management skills.
""")

section = st.sidebar.radio(
    "Navigation",
    [
        "Project Overview",
        "Data Cleaning",
        "EDA",
        "Forecasting",
        "Advanced Analysis",
        "Model Metrics",
        "Final Insights"
    ]
)


def show_image(path, caption):
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.warning(f"Missing file: {path}")


if section == "Project Overview":
    st.header("Project Overview")

    st.write("""
    This project analyzes the Global Weather Repository dataset to forecast weather trends.
    It includes data cleaning, exploratory data analysis, forecasting models, model comparison,
    anomaly detection, climate analysis, air-quality analysis, feature importance, and geographical analysis.
    """)

    if os.path.exists("outputs/cleaned_weather_data.csv"):
        df = pd.read_csv("outputs/cleaned_weather_data.csv")
        st.subheader("Cleaned Dataset Preview")
        st.dataframe(df.head())
        st.write("Rows:", df.shape[0])
        st.write("Columns:", df.shape[1])
    else:
        st.warning("Run main.py first to generate outputs.")


elif section == "Data Cleaning":
    st.header("Data Cleaning Summary")

    if os.path.exists("outputs/cleaning_summary.csv"):
        summary = pd.read_csv("outputs/cleaning_summary.csv")
        st.dataframe(summary)
    else:
        st.warning("Cleaning summary not found.")

    show_image("outputs/figures/missing_values.png", "Missing Values")


elif section == "EDA":
    st.header("Exploratory Data Analysis")

    show_image("outputs/figures/temperature_trend.png", "Temperature Trend")
    show_image("outputs/figures/precipitation_trend.png", "Precipitation Trend")
    show_image("outputs/figures/correlation_heatmap.png", "Correlation Heatmap")
    show_image("outputs/figures/top_countries.png", "Top Countries")


elif section == "Forecasting":
    st.header("Forecasting Results")

    show_image("outputs/figures/daily_temperature_forecast_comparison.png", "Daily Temperature Forecast Comparison")
    show_image("outputs/figures/feature_importance.png", "Feature Importance")


elif section == "Advanced Analysis":
    st.header("Advanced Analysis")

    st.subheader("Anomaly Detection")
    show_image("outputs/figures/anomaly_detection.png", "Anomaly Detection")

    st.subheader("Climate Analysis")
    show_image("outputs/figures/climate_monthly_patterns.png", "Monthly Climate Patterns")

    st.subheader("Air Quality Analysis")
    show_image("outputs/figures/air_quality_correlation_heatmap.png", "Air Quality Correlation")

    st.subheader("Geographical Analysis")
    show_image("outputs/figures/geographical_temperature_patterns.png", "Geographical Temperature Patterns")


elif section == "Model Metrics":
    st.header("Model Metrics")

    if os.path.exists("outputs/model_metrics.csv"):
        metrics = pd.read_csv("outputs/model_metrics.csv")
        st.dataframe(metrics)

        best_model = metrics.sort_values(by="RMSE", ascending=True).iloc[0]
        st.success(f"Best model by RMSE: {best_model['model']}")
    else:
        st.warning("Model metrics not found.")


elif section == "Final Insights":
    st.header("Final Insights")

    st.write("""
    In this section, I summarized the main results I found after running the full weather
    forecasting pipeline. I focused on model performance, feature importance, anomalies,
    climate patterns, air-quality relationships, and geographical differences.
    """)

    # ----------------------------
    # Forecasting insights
    # ----------------------------
    st.subheader("1. Forecasting Model Performance")

    if os.path.exists("outputs/model_metrics.csv"):
        metrics = pd.read_csv("outputs/model_metrics.csv")

        st.dataframe(metrics)

        best_model = metrics.sort_values(by="RMSE", ascending=True).iloc[0]
        ensemble_row = metrics[metrics["model"] == "Ensemble"].iloc[0]
        linear_row = metrics[metrics["model"] == "Linear Regression"].iloc[0]
        naive_row = metrics[metrics["model"] == "Naive Baseline"].iloc[0]
        moving_avg_row = metrics[metrics["model"] == "Moving Average"].iloc[0]

        st.success(
            f"The best model in my run was **{best_model['model']}**, "
            f"with an RMSE of **{best_model['RMSE']:.3f}**, "
            f"MAE of **{best_model['MAE']:.3f}**, "
            f"MAPE of **{best_model['MAPE']:.2f}%**, "
            f"and R² of **{best_model['R2']:.3f}**."
        )

        st.write(f"""
        From the model comparison, **Linear Regression gave the best overall result**.
        It had the lowest RMSE (**{linear_row['RMSE']:.3f}**) and the highest R²
        (**{linear_row['R2']:.3f}**). This was interesting because I expected the tree-based
        models to perform better, but in this case the simpler model worked best.

        The **Ensemble model also performed well**, with an RMSE of **{ensemble_row['RMSE']:.3f}**
        and R² of **{ensemble_row['R2']:.3f}**. However, it did not beat Linear Regression.
        This shows that combining models does not always guarantee the best result.

        The **Naive Baseline** had an RMSE of **{naive_row['RMSE']:.3f}**, while Linear Regression
        reduced the RMSE to **{linear_row['RMSE']:.3f}**. This tells me that the lag features,
        rolling averages, and calendar features helped the model learn useful temperature patterns.

        The **Moving Average model** also performed surprisingly well, with an RMSE of
        **{moving_avg_row['RMSE']:.3f}**. That makes sense because temperature usually changes
        gradually, so recent average temperature is a strong simple predictor.
        """)

    else:
        st.warning("Model metrics file not found. Run main.py first.")

    # ----------------------------
    # Feature importance insights
    # ----------------------------
    st.subheader("2. Feature Importance")

    if os.path.exists("outputs/feature_importance.csv"):
        importance = pd.read_csv("outputs/feature_importance.csv")
        st.dataframe(importance.head(10))

        top_feature = importance.iloc[0]

        st.write(f"""
        Based on the Random Forest feature importance results, the most important feature was
        **{top_feature['feature']}**. This means that this variable contributed the most to the
        Random Forest model's temperature predictions.

        In general, I expected lag features and rolling averages to be important because weather
        conditions are usually connected to recent weather. For example, today's temperature is
        often close to yesterday's temperature or the average temperature from the past few days.
        """)
    else:
        st.info("Feature importance output not found.")

    # ----------------------------
    # Anomaly insights
    # ----------------------------
    st.subheader("3. Anomaly Detection")

    if os.path.exists("outputs/anomaly_summary.csv"):
        anomalies = pd.read_csv("outputs/anomaly_summary.csv")

        st.write(f"""
        The anomaly detection step found **{len(anomalies)} unusual weather records**.
        I treated these as records that may represent extreme weather, unusual combinations
        of weather conditions, or possible data issues.

        I included anomaly detection because weather datasets can contain sudden spikes in
        temperature, precipitation, wind speed, or pressure. These unusual values are useful
        to review because they can affect both EDA and model performance.
        """)

        st.dataframe(anomalies.head(10))
    else:
        st.info("Anomaly summary output not found.")

    # ----------------------------
    # Climate insights
    # ----------------------------
    st.subheader("4. Climate and Seasonal Patterns")

    if os.path.exists("outputs/climate_summary.csv"):
        climate = pd.read_csv("outputs/climate_summary.csv")
        st.dataframe(climate.head(10))

        st.write("""
        For the climate analysis, I grouped the data by month and country to look at broader
        weather patterns instead of only daily changes. This helped me compare average
        temperature, precipitation, humidity, and wind speed across different months and locations.

        This part of the project is useful because the dataset contains global weather records.
        A single global average can hide regional differences, so grouping by country and month
        gives a clearer view of seasonal and geographical patterns.
        """)
    else:
        st.info("Climate summary output not found.")

    # ----------------------------
    # Air quality insights
    # ----------------------------
    st.subheader("5. Air Quality and Weather Relationships")

    if os.path.exists("outputs/air_quality_weather_correlations.csv"):
        air_corr = pd.read_csv(
            "outputs/air_quality_weather_correlations.csv",
            index_col=0
        )

        st.dataframe(air_corr)

        # Convert the correlation matrix into a simple table so it is easier to sort
        corr_pairs = air_corr.abs().stack().reset_index()
        corr_pairs.columns = [
            "weather_variable",
            "air_quality_variable",
            "abs_correlation"
        ]

        corr_pairs = corr_pairs.sort_values(
            by="abs_correlation",
            ascending=False
        )

        if not corr_pairs.empty:
            strongest_row = corr_pairs.iloc[0]

            weather_var = strongest_row["weather_variable"]
            air_quality_var = strongest_row["air_quality_variable"]

            # Get the original correlation value, not the absolute value
            corr_value = air_corr.loc[weather_var, air_quality_var]

            st.write(f"""
            The strongest relationship I found between a weather variable and an air-quality
            variable was between **{weather_var}** and **{air_quality_var}**, with a correlation
            of **{corr_value:.3f}**.

            I am not treating this as proof that one variable causes the other. It only shows
            that these two variables moved together in the dataset. Still, it gives a useful
            starting point for understanding how weather conditions may be related to air quality.
            """)
    else:
        st.info("Air-quality correlation output not found.")

    # ----------------------------
    # Geographical insights
    # ----------------------------
    st.subheader("6. Geographical Weather Patterns")

    if os.path.exists("outputs/geographical_summary.csv"):
        geography = pd.read_csv("outputs/geographical_summary.csv")
        st.dataframe(geography.head(10))

        if "country" in geography.columns and "temperature_celsius" in geography.columns:
            warmest_country = geography.sort_values(by="temperature_celsius", ascending=False).iloc[0]
            coolest_country = geography.sort_values(by="temperature_celsius", ascending=True).iloc[0]

            st.write(f"""
            The geographical analysis showed that weather patterns were different across countries.
            In this dataset, **{warmest_country['country']}** had one of the highest average
            temperatures, while **{coolest_country['country']}** had one of the lowest average
            temperatures.

            This was an important part of the analysis because the dataset is global. Weather
            patterns can look very different depending on the country or region, so it is helpful
            to compare locations instead of only looking at one overall average.
            """)
        else:
            st.write("""
            The geographical summary helped me compare weather conditions across countries.
            This gives more context to the global weather trends and shows how weather patterns
            vary by location.
            """)
    else:
        st.info("Geographical summary output not found.")

    # ----------------------------
    # Final conclusion
    # ----------------------------
    st.subheader("7. Overall Conclusion")

    st.write("""
    Overall, this project showed me that daily average temperature can be forecasted fairly well
    using the `last_updated` column, lag features, rolling averages, and calendar-based features.

    My best result came from Linear Regression. This was a useful finding because it showed that
    a simpler model can sometimes perform better than more complex models when the features are
    strong and the pattern is mostly linear.

    The Ensemble model was still useful because it performed better than several individual models,
    but it did not outperform Linear Regression in this run. This helped me understand that model
    comparison is important instead of assuming that the most complex model will always be best.

    The advanced analysis gave extra context beyond forecasting. Anomaly detection helped identify
    unusual records, climate analysis showed monthly and country-level patterns, air-quality analysis
    showed possible relationships between weather and pollution variables, and geographical analysis
    showed how weather differs across countries.
    """)