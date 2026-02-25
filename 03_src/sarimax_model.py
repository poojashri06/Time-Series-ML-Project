# Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
time1 = pd.read_csv(r"C:\Users\ramya\Downloads\time_series_with_external_factors.csv")
time1=pd.read_csv("feature_engineered_time_series.csv")
time1['date']=pd.to_datetime(time1['date'])
time1=time1.set_index('date')
time1.head()
# Select Target and Exogenous Variables
target = time1['electricity_demand']

exog_features = [
    'temperature_celsius',
    'rainfall_mm',
    'is_holiday',
    'is_weekend',
    'rolling_7_mean',
    'rolling_30_mean']

exog = time1[exog_features]
# Train-Test Split ( Time Based )
split_date = '2023-01-01'

Y_train = target[target.index < split_date]
Y_test  = target[target.index >= split_date]

exog_train = exog[exog.index < split_date]
exog_test  = exog[exog.index >= split_date]
# To Find Auto ARIMA
from pmdarima import auto_arima
arima = auto_arima(Y_train,seasonal=True,m=7)
print(arima)

# Build SARIMAX Model
model = SARIMAX(
    Y_train,
    exog=exog_train,
    order=(0, 1, 1),
    seasonal_order=(0, 0, 2, 7),
    enforce_stationarity=False,
    enforce_invertibility=False)
# Train the Model
sarimax_result = model.fit(disp=False)
sarimax_result.summary()
# Forecast Using External Factors
Y_pred = sarimax_result.predict(start=Y_test.index[0],end=Y_test.index[-1],exog=exog_test)
# Evaluate Model Performance
mae = mean_absolute_error(Y_test, Y_pred)
rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))

print("SARIMAX MAE:", mae)
print("SARIMAX RMSE:", rmse)
# Forecast Future Values 
future_steps = 30
future_exog = exog_test.iloc[:future_steps]
future_forecast = sarimax_result.forecast(
    steps=future_steps,
    exog=future_exog)

future_forecast
