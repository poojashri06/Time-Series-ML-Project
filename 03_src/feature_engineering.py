import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import math
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
time = pd.read_csv(r"C:\Users\ramya\Downloads\time_series_with_external_factors.csv")
time['date'] = pd.to_datetime(time['date'])
time = time.set_index('date')
# lag features created
time['lag_1'] = time['electricity_demand'].shift(1)
time['lag_7'] = time['electricity_demand'].shift(7)
time['lag_30'] = time['electricity_demand'].shift(30)

time[['electricity_demand','lag_1','lag_7','lag_30']].head(35)
time['rolling_7_mean'] = time['electricity_demand'].rolling(window=7).mean()
time['rolling_30_mean'] = time['electricity_demand'].rolling(window=30).mean()

time[['rolling_7_mean','rolling_30_mean']].head(40)
time['day'] = time.index.day
time['month'] = time.index.month
time['day_of_week'] = time.index.dayofweek
time['week_of_year'] = time.index.isocalendar().week

time[['day','month','day_of_week','week_of_year']].head()
# Createing weekend feature
time['is_weekend'] = np.where(time['day_of_week'] >= 5, 1, 0)
time[['day_of_week','is_weekend']].head(20)

# Handle missing values
time.isnull().sum()
# Drop Missing Rows

time = time.dropna()
time.isna().sum()

# Feature vs Target Separation

X = time.drop('electricity_demand', axis=1)
Y = time['electricity_demand']

print("Feature shape:", X.shape)
print("Target shape:", Y.shape)
X # it will only contains feature variables
Y # it will only can target variables
X.columns # Final Feature List
time.to_csv("feature_engineered_time_series.csv")
