import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
import numpy as np

# Filling data path
data_path_train = "./data//train.csv"
data_path_test = "./data//test.csv"

# reading CSV file
df_train = pd.read_csv(data_path_train)
df_test = pd.read_csv(data_path_test)

# Sorting by the id and resetting the index
df=df.sort_values("id").reset_index(drop=True)