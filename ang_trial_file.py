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

# # Dropping orientation (argue saying that this is hardly inputer and has a 30% of missing data) 
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

    # Replacing values below the lower limit with the lower limit
    winsorized_data[winsorized_data < lower_limit] = lower_limit

    # Replacing values above the upper limit with the upper limit
    winsorized_data[winsorized_data > upper_limit] = upper_limit

    return winsorized_data

# Windsorizing the price variable
df['price'] = winsorize(df['price'], limits=(0.05, 0.05))

# Replacing the outliers with NaN in the number of rooms (justify cutoff value: outliers are very high above 10)
df['num_rooms'] = df['num_rooms'].apply(lambda x: x if x<4 else np.nan)

# df['orientation'] = df['orientation'].apply(lambda x: x if x!= 'soxth' else 'south')

# Replacing the values of square metres < 40 with NaN (change the cutoff value and see the results)
df.loc[df['square_meters'] < 0, 'square_meters'] = np.nan

# Saving the value of floor 1 standardized
floor1_std = (1 - np.mean(df['floor'])) / np.std(df['floor'])

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

# Dropping NaNs from year built (justify: difficult to predict based on other variables, low value)
# df = df.dropna(subset=['year_built'])
df['year_built'].fillna(df['year_built'].mean())

# dropping the rows that have multiple missing values
cols = ['square_meters', 'floor', 'num_crimes']
cols1 = ['square_meters', 'floor', 'num_crimes']

for i in cols:
    for j in cols1:
        if i != j:
            df = df[(df[i].notnull()) | (df[j].notnull())]

###############
# Imputations #
###############

# Imputing with KNN
knn_cols = ['square_meters', 'floor', 'num_crimes', 'price']
df_sub = df[knn_cols]
imputer = KNNImputer(n_neighbors=20)
imputed_data = imputer.fit_transform(df_sub)
df_sub = pd.DataFrame(imputed_data, columns=df_sub.columns)
# print(round((df_sub.isnull().sum() / len(df_sub) * 100), 2))

# Putting the imputed columns back in the original df
df = df.reset_index(drop=True)
df = df.drop(knn_cols, axis=1)
df[knn_cols] = df_sub[knn_cols]
# print(round((df.isnull().sum() / len(df) * 100), 2))

# # Imputing room numbers
df['sqm_per_room'] = df['square_meters']/df['num_rooms']
Q1 = df['sqm_per_room'].quantile(0.25)
Q3 = df['sqm_per_room'].quantile(0.75)
IQR = Q3 - Q1

# Define the lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

median_sqm_per_room = df['sqm_per_room'].median()

def changing_num_rooms_in_outliers(row):
    if row['sqm_per_room'] < upper_bound:
        return row['num_rooms']
    else:
        return round(row['square_meters'] / median_sqm_per_room,1)
    
df['num_rooms'] = df.apply(changing_num_rooms_in_outliers, axis=1)


#################################
# Categorical Variable Encoding # 
#################################

# Using crime to order the neighborhoods by mean num_crimes
neighb_mean_crime = df.groupby('neighborhood')['num_crimes'].mean()
df['neighborhood_crime_encoded'] = df['neighborhood'].map(neighb_mean_crime)

#######################
# Feature Engineering # 
#######################

# Creating the floor1 dummy variable
df['floor_one_dummy'] = df['floor'].apply(lambda x: True if x==floor1_std else False)

#####################
# Orientation Dummy #
#####################

# df['south_orientation'] = df['orientation'].apply(lambda x: True if x=='south' else False)
# df.drop('orientation', axis=1, inplace=True)

###############################
# Dropping the remaining NaNs #
###############################
df = df.dropna(axis= 0)

######################
# Correlation matrix # 
######################

# Correlation matrix
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
correlation_matrix = df[numeric_columns].corr()

########################
# Linear interpolation #
########################

# df = df.sort_values('square_meters')
# numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
# correlation_matrix = df[numeric_columns].corr()

####################
# Trial Prediction #
####################

# features = ['num_rooms', 'num_baths','square_meters', 'floor', 'num_crimes', 'neighborhood_crime_encoded','has_ac', 'floor_one_dummy']
# target = ['price']

# def prediction_accuracy(df, features, target):
#     mse_list = []
#     num_of_predictions = 5000
#     for i in range (num_of_predictions):
#         X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size= 0.2)

#         model = LinearRegression()
#         model.fit(X_train, y_train)

#         y_pred = model.predict(X_test)
#         mse = mean_squared_error(y_test, y_pred)
#         mse_list.append(mse)
#     return sum(mse_list) / len(mse_list)

# print(prediction_accuracy(df, features, target))


##############
# Submitting #
##############

# ######################
# # OLLIE MODIFICATION #
# ######################

# Also train model without the binary variables
df_no_binary = df[['num_rooms', 'num_baths', 'square_meters', 'year_built', 'floor', 'num_crimes', 'neighborhood_crime_encoded', 'floor_one_dummy']]

y_train = df['price']
x_train = df_no_binary

model_no_binary = LinearRegression()
model_no_binary.fit(x_train, y_train)

# ####################
# # Linear Modelling #
# ####################

# Running simple linear model without feature scaling, using  num_crimes and square_meters as predictors
y_train = df['price']
x_train = df[['num_rooms', 'num_baths', 'square_meters', 'year_built', 'floor', 'num_crimes', 'neighborhood_crime_encoded', 'is_furnished', 'has_pool', 'num_crimes', 'has_ac', 'accepts_pets', 'floor_one_dummy']]

model = LinearRegression()
model.fit(x_train, y_train)

# Test data import
data_path_test = "./data//test.csv"
df_test = pd.read_csv(data_path_test)
df_test=df_test.sort_values("id").reset_index(drop=True)

# Dropping columns with high missing percentage
df_test.drop('num_supermarkets', axis=1, inplace=True)
df_test.drop('orientation', axis=1, inplace=True)

# Creating floor variable
df_test[['floor', 'door_num']] = df_test['door'].str.split('-', n=1, expand=True)
df_test['floor'] = df_test['floor'].str[0]
df_test["floor"] = pd.to_numeric(df_test["floor"])

# Dropping door and door_num columns
df_test.drop('door', axis=1, inplace=True)
df_test.drop('door_num', axis=1, inplace=True)

# Turning to nan sq.mt and num_rooms
df_test['num_rooms'] = df_test['num_rooms'].apply(lambda x: x if x<4 else np.nan)
df_test.loc[df_test['square_meters'] < 0, 'square_meters'] = np.nan

# Creation of a dummy variable for floor 1
df_test['floor_one_dummy'] = df_test['floor'].apply(lambda x: True if x==1 else False)

# Neighborhood encoding and dropping categorical variable
neighb_mean_crime = df_test.groupby('neighborhood')['num_crimes'].mean()
df_test['neighborhood_crime_encoded'] = df_test['neighborhood'].map(neighb_mean_crime)
df_test.drop('neighborhood', axis=1, inplace=True)

# Standardizing
to_standardize = ['num_rooms', 'num_baths', 'square_meters', 'year_built', 'floor', 'num_crimes', 'neighborhood_crime_encoded']
for i in to_standardize:
    df_test[i] = (df_test[i] - np.mean(df_test[i])) / np.std(df_test[i])

# OLLIE NEW CODE: 
# Subsetting df for those which have a missing binary variable
binary_cols = ['is_furnished', 'has_pool', 'has_ac', 'accepts_pets']
df_missing = df_test[df_test[binary_cols].isna().any(axis=1)]
df_not_missing = df_test[~df_test[binary_cols].isna().any(axis=1)]

# Using KNN on not_missing df
columns_to_knn = ['id', 'num_baths', 'square_meters', 'year_built', 'is_furnished', 'has_pool', 'num_crimes', 'has_ac', 'accepts_pets', 'floor', 'floor_one_dummy', 'neighborhood_crime_encoded']
df_sub_nm = df_not_missing[columns_to_knn]
imputer = KNNImputer(n_neighbors=3)
imputed_data = imputer.fit_transform(df_sub_nm)
df_sub_nm = pd.DataFrame(imputed_data, columns=columns_to_knn)

df_not_missing = df_not_missing.reset_index(drop=True)
df_not_missing = df_not_missing.drop(columns_to_knn, axis=1)
df_not_missing[columns_to_knn] = df_sub_nm[columns_to_knn]

df_not_missing['sqm_per_room'] = df_not_missing['square_meters']/df_not_missing['num_rooms']
Q1 = df_not_missing['sqm_per_room'].quantile(0.25)
Q3 = df_not_missing['sqm_per_room'].quantile(0.75)
IQR = Q3 - Q1

# Define the lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

median_sqm_per_room = df_not_missing['sqm_per_room'].median()

def changing_num_rooms_in_outliers(row):
    if row['sqm_per_room'] < upper_bound:
        return row['num_rooms']
    else:
        return round(row['square_meters'] / median_sqm_per_room,1)
    
df_not_missing['num_rooms'] = df_not_missing.apply(changing_num_rooms_in_outliers, axis=1)
# Using info from the other dataframe to KNN this one
df_sub_m = df_missing[columns_to_knn]
imputed_data_missing = imputer.transform(df_sub_m)  # Use transform instead of fit_transform
df_sub_m = pd.DataFrame(imputed_data_missing, columns=columns_to_knn)

df_missing = df_missing.reset_index(drop=True)
df_missing = df_missing.drop(columns_to_knn, axis=1)
df_missing[columns_to_knn] = df_sub_m[columns_to_knn]

# Imputing the num of rooms
df_missing['sqm_per_room'] = df_missing['square_meters']/df_missing['num_rooms']
Q1 = df_missing['sqm_per_room'].quantile(0.25)
Q3 = df_missing['sqm_per_room'].quantile(0.75)
IQR = Q3 - Q1

# Define the lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

median_sqm_per_room = df_missing['sqm_per_room'].median()

def changing_num_rooms_in_outliers(row):
    if row['sqm_per_room'] < upper_bound:
        return row['num_rooms']
    else:
        return round(row['square_meters'] / median_sqm_per_room,1)
    
df_missing['num_rooms'] = df_missing.apply(changing_num_rooms_in_outliers, axis=1)

# Drop binaries from df_missing
df_missing.drop(binary_cols, axis=1, inplace=True)

# Prediction for df_not_missing
x_test = df_not_missing[['num_rooms', 'num_baths', 'square_meters', 'year_built', 'floor', 'num_crimes', 'neighborhood_crime_encoded', 'is_furnished', 'has_pool', 'num_crimes', 'has_ac', 'accepts_pets', 'floor_one_dummy']]
y_pred_not_missing = model.predict(x_test)

df_not_missing['pred'] = y_pred_not_missing

# Prediction for df_missing
x_test = df_missing[['num_rooms', 'num_baths', 'square_meters', 'year_built', 'floor', 'num_crimes', 'neighborhood_crime_encoded', 'floor_one_dummy']]
y_pred_missing = model_no_binary.predict(x_test)

df_missing['pred'] = y_pred_missing
new_df = pd.DataFrame()

# Creating final DataFrame
new_df['id'] = df_missing['id'].tolist() + df_not_missing['id'].tolist()
new_df['pred'] = df_missing['pred'].tolist() + df_not_missing['pred'].tolist()

new_df.to_csv('./prediction_sunday8pm.csv', index=False)
