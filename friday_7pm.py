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
data_path = "./data//train.csv"

# reading CSV file
df = pd.read_csv(data_path)

# Sorting by the id and resetting the index
df=df.sort_values("id").reset_index(drop=True)

#################################################
# Dropping columns with high missing percentage #
#################################################

missing_percentage = df.isnull().sum()/len(df) * 100

# Dropping supermarkets number
df.drop('num_supermarkets', axis=1, inplace=True)

# Dropping orientation (argue saying that this is hardly inputer and has a 30% of missing data) 
df.drop('orientation', axis=1, inplace=True)


###########################
# Creating floor variable #
###########################

# Creating the floor variable
df[['floor', 'door_num']] = df['door'].str.split('-', n=1, expand=True)
df['floor'] = df['floor'].str[0]
df["floor"] = pd.to_numeric(df["floor"])

# Dropping door and door_num columns (justify: not influential)
df.drop('door', axis=1, inplace=True)
df.drop('door_num', axis=1, inplace=True)

# The distribution of price to floor is interesting (the means are growing) - high floors (skyscrapers-cheaper)
# plt.scatter(df['floor'], df['price'])
# print(df.groupby('floor')['price'].mean())


#####################
# Handling outliers #
#####################

# Checking for outliers in price column
threshold=3.0
mean = np.mean(df['price'])
std = np.std(df['price'])
cutoff = threshold * std
lower_bound = mean - cutoff
upper_bound = mean + cutoff

# Calculating the number of records below and above lower and above bound value respectively
outliers = [x for x in df['price'] if (x >= upper_bound) | (x <= lower_bound)]

# Windsorizing price outliers
def winsorize(data, limits=(0.05, 0.05)):
    """
    Winsorize a dataset by replacing extreme values with less extreme values.

    Arguments:
    - data: 1-D array or list, the dataset to be winsorized.
    - limits: Tuple of two floats (lower, upper), representing the fraction of values to be replaced
              on each tail. Default is (0.05, 0.05), which replaces 5% of the values on each tail.

    Returns:
    - winsorized_data: 1-D array, the winsorized dataset.
    """
    # Copy the input data to avoid modifying the original array
    winsorized_data = np.copy(data)

    # Calculating the lower and upper limits for winsorization
    lower_limit = np.percentile(winsorized_data, limits[0] * 100)
    upper_limit = np.percentile(winsorized_data, 100 - limits[1] * 100)

    print('Lower limit:', lower_limit)
    print('Upper limit:', upper_limit)

    # Replacing values below the lower limit with the lower limit
    winsorized_data[winsorized_data < lower_limit] = lower_limit

    # Replacing values above the upper limit with the upper limit
    winsorized_data[winsorized_data > upper_limit] = upper_limit

    return winsorized_data

# Windsorizing the price variable
df['price'] = winsorize(df['price'], limits=(0.05, 0.05))

# Replacing the outliers with NaN in the number of rooms (justify cutoff value: outliers are very high above 10)
df['num_rooms'] = df['num_rooms'].apply(lambda x: x if x<4 else np.nan)

# Replacing the values of square metres < 40 with NaN (change the cutoff value and see the results)
df.loc[df['square_meters'] < 0, 'square_meters'] = np.nan

# Creation of a dummy variable for floor 1
df['floor_one_dummy'] = df['floor'].apply(lambda x: True if x==1 else False)

###################
# Standardization #
###################

to_standardize = ['num_rooms', 'num_baths', 'square_meters', 'year_built', 'floor', 'num_crimes']

for i in to_standardize:
    df[i] = (df[i] - np.mean(df[i])) / np.std(df[i])

#################
# Normalization #
#################

# to_normalize = ['num_rooms', 'num_baths', 'year_built', 'square_meters', 'floor', 'num_crimes']

# def normalize_data(data):
#     min_value = min(data.dropna())
#     max_value = max(data.dropna())
#     normalized_data = []

#     for value in data:
#         normalized_value = (value - min_value) / (max_value - min_value)
#         normalized_data.append(normalized_value)

#     return normalized_data

# for col in to_normalize:
#     df[col] = normalize_data(df[col])

#########################
# Handling missing data #
#########################

# Missing values percentage
# print(round((df.isnull().sum() / len(df) * 100), 2))

# Dropping NaNs from year built (justify: difficult to predict based on other variables, low value)
df = df.dropna(subset=['year_built'])

# dropping the rows that have multiple missing values
cols = ['num_rooms', 'num_baths', 'square_meters', 'year_built', 'floor', 'num_crimes']
cols1 = ['num_rooms', 'num_baths', 'square_meters', 'year_built', 'floor', 'num_crimes']

for i in cols:
    for j in cols1:
        if i != j:
            df = df[(df[i].notnull()) | (df[j].notnull())]

#####################
# Imputing with KNN #
#####################
knn_cols = ['num_rooms', 'num_baths', 'square_meters', 'floor', 'num_crimes', 'price']
df_sub = df[knn_cols]
imputer = KNNImputer(n_neighbors=7)
imputed_data = imputer.fit_transform(df_sub)
df_sub = pd.DataFrame(imputed_data, columns=df_sub.columns)
# print(round((df_sub.isnull().sum() / len(df_sub) * 100), 2))

# Putting the imputed columns back in the original df
df = df.reset_index(drop=True)
df = df.drop(knn_cols, axis=1)
df[knn_cols] = df_sub[knn_cols]
# print(round((df.isnull().sum() / len(df) * 100), 2))

# dropping the remaining NaNs
df = df.dropna(axis= 0)

#################################
# Categorical Variable Encoding # 
#################################

# Using crime to order the neighborhoods by mean num_crimes
# sns.boxplot(data=df, x='neighborhood', y='num_crimes')

neighb_mean_crime = df.groupby('neighborhood')['num_crimes'].mean()
df['neighborhood_crime_encoded'] = df['neighborhood'].map(neighb_mean_crime)

########################
# Feature Engiineering # 
########################

df['rooms_per_bath'] = df['num_rooms']/df['num_baths']

##################
# Splitting data #
##################

features = ['num_rooms', 'num_baths', 'square_meters', 'floor', 'num_crimes', 'neighborhood_crime_encoded','has_ac', 'floor_one_dummy', 'rooms_per_bath']
target = ['price']
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size= 0.2, random_state=42)

####################
# Linear Modelling #
####################

# Running simple linear model
model = LinearRegression()
model.fit(X_train, y_train)
# model.fit(df[features], df[target])

# Model res
model.score(df[features], df[target])

###########
# Testing #
###########

y_pred = model.predict(X_test)
r_squared = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(r_squared)
print(mse)


##############
# Submitting #
##############

# # Test data import
# data_path_test = "./data//test.csv"
# df_test = pd.read_csv(data_path_test)
# df_test=df_test.sort_values("id").reset_index(drop=True)

# # Dropping columns with high missing percentage
# df_test.drop('num_supermarkets', axis=1, inplace=True)
# df_test.drop('orientation', axis=1, inplace=True)

# # Creating floor variable
# df_test[['floor', 'door_num']] = df_test['door'].str.split('-', n=1, expand=True)
# df_test['floor'] = df_test['floor'].str[0]
# df_test["floor"] = pd.to_numeric(df_test["floor"])

# # Dropping door and door_num columns
# df_test.drop('door', axis=1, inplace=True)
# df_test.drop('door_num', axis=1, inplace=True)

# # Turning to nan sq.mt and num_rooms
# df_test['num_rooms'] = df_test['num_rooms'].apply(lambda x: x if x<4 else np.nan)
# df_test.loc[df_test['square_meters'] < 0, 'square_meters'] = np.nan

# # Neighborhood encoding and dropping categorical variable
# neighb_mean_crime = df_test.groupby('neighborhood')['num_crimes'].mean()
# df_test['neighborhood_crime_encoded'] = df_test['neighborhood'].map(neighb_mean_crime)
# df_test.drop('neighborhood', axis=1, inplace=True)

# # Standardizing
# to_standardize = ['num_rooms', 'num_baths', 'square_meters', 'year_built', 'floor', 'num_crimes', 'neighborhood_crime_encoded']
# for i in to_standardize:
#     df_test[i] = (df_test[i] - np.mean(df_test[i])) / np.std(df_test[i])

# df = df.dropna(subset=['year_built'])
# # KNN 
# imputer = KNNImputer(n_neighbors=3)
# imputed_data = imputer.fit_transform(df_test)
# df_test = pd.DataFrame(imputed_data, columns=df_test.columns)
# print(round((df_test.isnull().sum() / len(df_test) * 100), 2))

# # Prediction
# x_test = df_test[features]
# y_pred = model.predict(x_test)

# df_pred = pd.DataFrame()
# df_pred['id'] = df_test['id']
# df_pred['price'] = y_pred

# df_pred.to_csv('./prediction_friday7_30pm.csv', index=False)