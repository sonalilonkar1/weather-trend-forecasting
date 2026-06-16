# Weather Trend Forecasting Assessment

This repository is for the **Tech Assignment: Weather Trend Forecasting** solving Advanced Assignment.

## Project Overview

This project analyzes the Global Weather Repository dataset from Kaggle to explore global weather patterns and forecast daily average temperature trends.

For this assignment, I focused on building a complete data science workflow: cleaning the dataset, exploring temperature and precipitation patterns, creating a time-series forecasting pipeline using the last_updated column, comparing multiple models, and adding advanced analysis such as anomaly detection, air-quality correlation, feature importance, climate patterns, and geographical summaries.

## PM Accelerator Mission
By making industry-leading tools and education available to individuals from all backgrounds, we level the playing field for future PM leaders. This is the PM Accelerator motto, as we grant aspiring and experienced PMs what they need most – Access. 
We introduce you to industry leaders, surround you with the right PM ecosystem, and discover the new world of AI product management skills.

More info on: https://www.pmaccelerator.io/about-us and https://www.linkedin.com/school/pmaccelerator/

## Author

**Sonali Lonkar**  
AI Engineer Intern Technical Assessment  
Completed: Basic & Advanced Assessment - Tech Assessment: Weather Trend Forecasting

## Demo Video

[Watch the project demo video](https://drive.google.com/file/d/15iL0YQDuWIyF-KDuI5EFg-Ht_IB1MQ89/view?usp=sharing)

## Dataset

Download the dataset from Kaggle:

- Kaggle dataset: **Global Weather Repository / World Weather Repository**
- Expected local file name: `GlobalWeatherRepository.csv`

Place the CSV here:

```text
data/GlobalWeatherRepository.csv
```

## Project Structure

```text
weather_trend_forecasting_project/
├── app.py                                  # Streamlit dashboard
├── main.py                                 # End-to-end pipeline
├── requirements.txt                        # Python dependencies
├── .gitignore                              # gitignore
├── README.md                               # Detailed info about project and report
├── data/GlobalWeatherRepository.csv        # Kaggle CSV 
├── notebooks/01_exploration.ipynb          # Exploratory notebook 
├── outputs/                                # Generated CSVs and figures
├── reports/Final_report.pdf                # Final Project Report
└── src/
    ├── advanced_analysis.py                # Advanced analyses
    ├── data_cleaning.py                    # Loading, cleaning, preprocessing
    ├── eda.py                              # Exploration charts
    ├── forecasting.py                      # Forecasting models and ensemble
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Full Pipeline

```bash
python main.py --data data/GlobalWeatherRepository.csv
```
This script runs the full workflow:

Loads the raw dataset
Cleans and preprocesses the data
Creates EDA charts
Builds forecasting models
Evaluates model performance
Runs advanced analysis
Saves all outputs to the outputs/ folder

## Run the Dashboard

After running main.py, start the Streamlit dashboard: 

```bash
streamlit run app.py
```

### The dashboard includes:

Project overview
PM Accelerator mission
Data cleaning summary
EDA visualizations
Forecasting results
Model metrics
Feature importance
Anomaly detection
Climate analysis
Air-quality analysis
Geographical analysis
Final insights

## Outputs

After `main.py` runs, these files are created:

```text
outputs/air_quality_weather_correlations.csv
outputs/anomaly_summary.csv
outputs/cleaned_weather_data.csv
outputs/cleaning_summary.csv
outputs/climate_summary.csv
outputs/dataset_summary.csv
outputs/feature_importance.csv
outputs/geographical_summary.csv
outputs/model_metrics.csv
outputs/figures/air_quality_correlation_heatmap.png
outputs/figures/anomaly_detection.png
outputs/figures/climate_monthly_patterns.png
outputs/figures/correlation_heatmap.png
outputs/figures/feature_importance.png
outputs/figures/forecast_comparison.png
outputs/figures/geographical_temperature_patterns.png
outputs/figures/precipitation_trend.png
outputs/figures/temperature_trend.png
outputs/figures/top_countries.png
```

## Methodology

### 1. Data Cleaning and Preprocessing

I cleaned the dataset by:

- Standardizing column names
- Converting `last_updated` into a datetime column
- Removing duplicate records
- Filling missing numeric values with median values
- Filling missing categorical values with `Unknown`
- Removing rows with invalid dates
- Capping numeric outliers using percentile-based clipping
- Creating a cleaned dataset output for later analysis

The cleaned dataset is saved as:

`outputs/cleaned_weather_data.csv`

The cleaning summary is saved as:

`outputs/cleaning_summary.csv`

### 2. Exploratory Data Analysis

The EDA section focuses on understanding the dataset before modeling.

I created visualizations for:

- Average temperature trend over time
- Average precipitation trend over time
- Correlation between numeric weather variables
- Top countries by number of records
- Missing value patterns

Important chart outputs include:

`outputs/figures/temperature_trend.png`
`outputs/figures/precipitation_trend.png`
`outputs/figures/correlation_heatmap.png`
`outputs/figures/top_countries.png`

### 3. Forecasting Approach

For forecasting, I used `last_updated` as the time-series column and selected `temperature_celsius` as the main target variable.

I chose temperature because it is numeric, easy to interpret, and directly related to weather trend forecasting.

Since the dataset contains records from many locations, I aggregated the data into daily average temperature before modeling.

The forecasting features included:

- Previous day temperature
- 3-day lag temperature
- 7-day lag temperature
- 3-day rolling average
- 7-day rolling average
- Day of week
- Month

I used a chronological train/test split instead of a random split because this is a time-series problem. This prevents future data from leaking into the training set.

### 4. Models Compared

I compared the following models:

1. Naive Baseline
2. Moving Average
3. Linear Regression
4. Random Forest
5. Gradient Boosting
6. Ensemble Model

The Naive Baseline predicts using the previous temperature value. I included it because every forecasting model should be compared against a simple benchmark.

The Ensemble model averages predictions from multiple models. I included it to test whether combining models would improve forecast accuracy.

### 5. Model Evaluation Results

The models were evaluated using:

* MAE
* RMSE
* MAPE
* R²

My model results were:

| Model             |   MAE |  RMSE |  MAPE |    R² |
| ----------------- | ----: | ----: | ----: | ----: |
| Naive Baseline    | 0.393 | 1.304 | 3.75% | 0.764 |
| Moving Average    | 0.400 | 0.932 | 3.55% | 0.879 |
| Linear Regression | 0.225 | 0.668 | 1.95% | 0.938 |
| Random Forest     | 0.301 | 0.939 | 3.15% | 0.877 |
| Gradient Boosting | 0.289 | 0.936 | 3.06% | 0.878 |
| Ensemble          | 0.253 | 0.776 | 2.61% | 0.916 |

The best-performing model was **Linear Regression**, with:

* **MAE:** 0.225
* **RMSE:** 0.668
* **MAPE:** 1.95%
* **R²:** 0.938

This result was interesting because I originally expected the tree-based models or ensemble model to perform best. In this run, the simpler Linear Regression model performed better, which suggests that the lag features, rolling averages, and calendar features captured much of the temperature pattern.

The Ensemble model still performed well, but it did not outperform Linear Regression. This shows that a more complex model is not always better.

---

## Advanced Analysis

### Anomaly Detection

I used anomaly detection to identify unusual weather records. These records may represent extreme weather events, unusual combinations of weather variables, or possible data quality issues.

The anomaly output is saved as:

```text
outputs/anomaly_summary.csv
```

### Climate Analysis

I grouped the data by month and country to study broader climate and seasonal patterns. This helped compare temperature, precipitation, humidity, and wind speed across locations and time periods.

The climate summary is saved as:

```text
outputs/climate_summary.csv
```

### Air-Quality Analysis

I analyzed correlations between air-quality variables and weather variables such as temperature, humidity, precipitation, pressure, and wind speed.

The correlation output is saved as:

```text
outputs/air_quality_weather_correlations.csv
```

### Feature Importance

I used Random Forest feature importance to understand which engineered features contributed most to the prediction task.

The feature importance output is saved as:

```text
outputs/feature_importance.csv
```

### Geographical Analysis

I grouped the data by country to compare weather patterns across different locations. This helped show that global weather trends can hide important regional differences.

The geographical summary is saved as:

```text
outputs/geographical_summary.csv
```

---

## Key Insights

* Linear Regression was the best model in this run, with the lowest RMSE and highest R².
* The Ensemble model performed well, but it did not beat Linear Regression.
* The Moving Average model also performed strongly, which makes sense because temperature usually changes gradually.
* Lag features and rolling averages were useful for forecasting daily average temperature.
* Comparing against a Naive Baseline helped show that the forecasting features improved model performance.
* Anomaly detection helped identify unusual weather records that may be worth reviewing separately.
* Climate and geographical analysis showed that weather patterns vary across countries and months.
* Air-quality correlation analysis gave additional context about how pollution indicators may relate to weather variables.

---

## Limitations

There are a few limitations in this project:

* The main forecast uses global daily average temperature, so it does not capture detailed city-level forecasting.
* The dataset combines many countries and cities, which can smooth out local weather patterns.
* The model does not include external climate factors such as elevation, ocean patterns, or seasonal climate indexes.
* Correlation analysis does not prove causation.
* The forecasting models are simple and interpretable, but more advanced time-series methods could be tested in future work.

---

## Future Improvements

Future improvements could include:

* Forecasting temperature for individual cities or countries
* Adding deep learning models
* Creating more interactive geographical maps
* Adding more detailed air-quality analysis
* Comparing forecasts across different regions
* Improving feature engineering with seasonal and location-based variables
* Adding model saving and prediction scripts for future use

---

## How to Reproduce My Results

1. Clone the repository.
2. Create a virtual environment.
3. Install dependencies using `requirements.txt`.
4. Download the Kaggle dataset.
5. Place the CSV file in the `data/` folder.
6. Run:

```bash
python main.py --data data/GlobalWeatherRepository.csv
```

7. Open the dashboard:

```bash
streamlit run app.py
```

---

## License

This project is open source and available under the [MIT License](LICENSE).
