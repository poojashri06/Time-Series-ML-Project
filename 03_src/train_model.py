import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
time1 = pd.read_csv(r"feature_engineered_time_series.csv")
time1.head()
time1['date'] = pd.to_datetime(time1['date'])
time1 = time1.set_index('date')
# Define Features and Target
X = time1.drop('electricity_demand', axis=1)
Y = time1['electricity_demand']
time1.head()
split_date = '2023-01-01'

X_train = X[X.index < split_date]
X_test  = X[X.index >= split_date]

Y_train = Y[Y.index < split_date]
Y_test  = Y[Y.index >= split_date]

print(X_train.shape, X_test.shape)
# Base Line Model - Linear Regression
lr=LinearRegression()
lr.fit(X_train,Y_train)
Y_pred_lr=lr.predict(X_test)
# Evaluate Linear Regression 
mae_lr = mean_absolute_error(Y_test, Y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(Y_test, Y_pred_lr))

print("Linear Regression MAE:", mae_lr)
print("Linear Regression RMSE:", rmse_lr)
# Plot Predictions-Linear Regression 
plt.figure(figsize=(12,5))
plt.plot(Y_test.index, Y_test, label="Actual")
plt.plot(Y_test.index, Y_pred_lr, label="Predicted")
plt.title("Linear Regression: Actual vs Predicted")
plt.legend()
plt.show()
# Advanced Model - Random Forest Regressor
rfc = RandomForestRegressor(n_estimators=100,max_depth=10,random_state=42)
rfc.fit(X_train, Y_train)
Y_pred_rfc = rfc.predict(X_test)
# Evaluate Linear Regression 
mae_lr = mean_absolute_error(Y_test, Y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(Y_test, Y_pred_lr))

print("Linear Regression MAE:", mae_lr)
print("Linear Regression RMSE:", rmse_lr)
