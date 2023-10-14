import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.impute import KNNImputer
import numpy as np

path_test = './data/test.csv'
path_sample_submission = './data/sample_submission.csv'
data_test = pd.read_csv(path_test)

df = pd.DataFrame(data_test)
df.drop('num_supermarkets', axis=1, inplace=True)
df['num_rooms'] = df['num_rooms'].apply(lambda x: x if x<10 else np.nan)
df = df[(df['square_meters'] > 0) | (df['square_meters'].isna())]
norm_cols = ['num_rooms','num_baths','square_meters','year_built','num_crimes']
df_norm = df[norm_cols]

def normalize_data(data):
    min_value = min(data.dropna())
    max_value = max(data.dropna())
    normalized_data = []

    for value in data:
        normalized_value = (value - min_value) / (max_value - min_value)
        normalized_data.append(normalized_value)

    return normalized_data


for col in norm_cols:
    df_norm[col] = normalize_data(df_norm[col])
imputer = KNNImputer(n_neighbors=2)
imputed_data = imputer.fit_transform(df_norm)
imputed_df = pd.DataFrame(imputed_data, columns=df_norm.columns)

imputed_df = imputed_df.add_prefix('norm_')

df.reset_index(drop=True, inplace=True)
imputed_df.reset_index(drop=True, inplace=True)
df = pd.concat([df, imputed_df], axis=1)
