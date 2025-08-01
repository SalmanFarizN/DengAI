## DengAI competition

Models Attempted: LinearRegression, XGBRegressor, NegativeBinomial (baseline model), SARIMAX.

Feature Selection: Most correlated features with the target, all features.

Feautre Engineering: Sin and Cos of weekofyear to account for seasonlity in XGBoost, features with lag (t-1)...(t-5).
Saturation Deficit: Use dew point and air temperature to calculate how close the air is to saturation:
`saturation_deficit = reanalysis_air_temp_k - reanalysis_dew_point_temp_k`
Temperature-Based Indicators Temperature Suitability Index: Create a composite score based on how close temperatures are to optimal mosquito development ranges (typically 25-30°C). You could use: `temp_suitability = 1 - abs(station_avg_temp_c - 27.5) / 27.5`

The best performing model: XGBoost with top 3 correlated variables and their lags (t-1)...(t-3) with the sin/cos of the weekofday.
Score: MAE=25.0409
