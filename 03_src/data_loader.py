import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import math
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
ts=pd.read_csv(r"C:\Users\pooja\OneDrive\Documents\Chinna projects\time_series_with_external_factors.csv")
ts['date']=pd.to_datetime(ts['date'])
ts=ts.set_index('date')
