# Time-Series Forecasting of Carbon Monoxide and Nitrogen Dioxide Levels
This repository implements advanced time-series models to predict the future concentrations of major atmospheric pollutants, Carbon Monoxide (CO) and Nitrogen Dioxide (NO₂). The goal is to provide a reliable short-term forecast for air quality management and public health initiatives.

### Project Overview

This project aims to build predictive models to forecast daily or hourly concentrations of **CO(GT)** and **NO₂(GT)** based on historical data. The core objective is to move from reactive air quality assessment to proactive forecasting, enabling authorities to issue timely warnings and implement data-driven pollution control measures.

The pipeline involves comprehensive data cleaning, exploratory analysis of temporal patterns (trend, seasonality), model selection, hyperparameter tuning, and rigorous performance evaluation against real-world metrics like Mean Absolute Error (MAE).


### Methodology

1️⃣ Data Preprocessing:- 
                        Handling missing values (Imputation/Interpolation)
                        Outlier detection and smoothing
                        Feature Scaling/Normalization
                        Stationarity check (ADF Test) and differencing

2️⃣ Exploratory Data Analysis (EDA):-
                        Decomposition of time series (Trend, Seasonality, Residual)
                        Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) plots
                        Correlation analysis with meteorological factors

3️⃣ Models Trained:-
                    Seasonal ARIMA/SARIMAX (Statistical Baseline)
                    Facebook Prophet (Handling multiple seasonalities)
                    Long Short-Term Memory (LSTM) Networks (Deep Learning Approach)

4️⃣ Model Tuning:-
                  Auto-ARIMA for optimal (p, d, q) parameters
                  Bayesian Optimization for LSTM learning rate and hidden units

5️⃣ Evaluation Metrics:-
                        Mean Absolute Error (MAE)
                        Root Mean Squared Error (RMSE)
                        Mean Absolute Percentage Error (MAPE)

📊 Key Results

  Best Model: SARIMAX (for CO) and LSTM (for NO₂)
  Best 24-Hour MAE (CO): ~0.05 mg/m³
  Best 24-Hour MAE (NO₂): ~3.2 µg/m³
  Forecast Window: 14 days

### Insights:-

CO levels are observed to be higher on weekdays, suggesting a strong correlation with rush-hour traffic and human activity.
NO₂ also tends to be higher on weekdays, indicating a strong link to traffic and industrial/office area emissions.
The 14-day forecast suggested low immediate risk, with CO and NO₂ levels expected to remain below the high-risk threshold for the forecast period.

### Challenges Faced:-

Handling non-stationarity and multiple seasonality (daily, weekly, yearly) simultaneously was complex.
Input feature selection (exogenous variables like temperature/wind speed) proved crucial for SARIMAX performance.
LSTM training required significant computational resources and careful sequence-length selection.
Accurate long-term forecasting (beyond 7 days) was challenging due to the inherent volatility of atmospheric data.

### Key Learnings:-

Statistical models (SARIMAX) often provide a strong, interpretable baseline for time-series forecasting.
Deep learning models (LSTM) excel at capturing non-linear relationships and dependencies in multivariate air quality data.
Domain knowledge (e.g., traffic patterns, industrial activity) is essential for effective feature engineering and model interpretation.
Evaluation metrics like MAE are more robust than RMSE for reporting public health impact, as they penalize large errors linearly.

### Public Health Recommendations (Based on Forecast)

Public health advice was generated, recommending that vulnerable groups (children, elderly, asthma/COPD patients) limit outdoor exposure, use masks, and improve indoor ventilation on high-risk days.
Stricter monitoring and real-time data–driven measures (speed limits, entry restrictions) should be applied around traffic hotspots and industrial belts.

### Tech Stack:-

Language: Python
Libraries: pandas, numpy, statsmodels, Prophet, tensorflow/keras (for LSTM), scikit-learn, matplotlib, seaborn
Tools: Jupyter Notebook

### Author

Sojib Roy

If you find this project useful, consider giving it a star!
