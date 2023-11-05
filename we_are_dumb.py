import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Set data paths
train_data_path = "./data//train.csv"
test_data_path = "./data//test.csv"

# Reading data
df_train = pd.read_csv(train_data_path)
df_test = pd.read_csv(test_data_path)


# Windsorizing price outliers in train data
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
df_train['price'] = winsorize(df_train['price'].dropna(), limits=(0.05, 0.05))

# Merging dataframes
df = pd.concat([df_train, df_test], axis=0).sort_values("id").reset_index()


# Dropping orientation (argue saying that this is hardly inputer and has a 30% of missing data) 
df.drop('orientation', axis=1, inplace=True)

####CHANGED CODE############
df['floor'] = df['door'].str.extract(r'(\d+)º').astype(float)
# Dropping door and door_num columns (justify: not influential)
df.drop('door', axis=1, inplace=True)
#############################

# Replacing the outliers with NaN in the number of rooms (justify cutoff value: outliers are very high above 10)
df['num_rooms'] = df['num_rooms'].apply(lambda x: x if x<10 else np.nan)

# Replacing the values of square metres < 40 with NaN (change the cutoff value and see the results)
df.loc[df['square_meters'] < 0, 'square_meters'] = np.nan


# Feature engineering - dummy for floor 1
df['floor_one_dummy'] = df['floor'].apply(lambda x: True if x==1 else False)


# Standardization
to_standardize = ['square_meters', 'year_built','num_crimes']

for i in to_standardize:
    df[i] = (df[i] - np.mean(df[i])) / np.std(df[i])
    

# Encoding neighborhood
neighb_mean_crime = df.groupby('neighborhood')['num_crimes'].mean()
df['neighborhood_crime_encoded'] = df['neighborhood'].map(neighb_mean_crime)


# Spliting dataframes before dropping
df_test = df[df['price'].isna()]
df_train = df[df['price'].notna()]


# Dropping those with yr built missing in train data
df_train = df_train.dropna(subset=['year_built'])


# Dropping the rows that have multiple missing values for train data
cols = ['num_rooms', 'num_baths', 'square_meters', 'year_built', 'floor', 'num_crimes']
cols1 = ['num_rooms', 'num_baths', 'square_meters', 'year_built', 'floor', 'num_crimes']

for i in cols:
    for j in cols1:
        if i != j:
            df_train = df_train[(df_train[i].notnull()) | (df_train[j].notnull())]
            



# Imputing with linear regression 


# Imputing different combinations of variables with what makes most sense in train data


plt.figure(figsize=(8, 6))
sns.heatmap(df_train[['num_rooms', 'num_baths', 'square_meters', 'year_built', 'is_furnished', 'has_pool', 'num_crimes', 'has_ac', 'accepts_pets', 'price', 'floor']].corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()

# THIS FUNCTION NEEDS TO BE CHANGED, DOESN'T MAKE SENSE AT THE MOMENT - you always need to have cols_to_impute and predictor_cols equal
# Imputing num_rooms
def reg_imputer(df, columns_to_impute: list, predictor_columns: list):

    # Separate the DataFrames
    imputation_df = df[columns_to_impute]
    predictors_df = df[predictor_columns]

    imputer = IterativeImputer(estimator=LinearRegression())
    imputer.fit(predictors_df)
    imputed_values = imputer.transform(imputation_df)
    df[columns_to_impute] = imputed_values
    
    return df

# Just imputing num_crimes and square_meters with price as these are the only ones with correlation
df_train = reg_imputer(df_train, ['square_meters', 'num_crimes', 'price'], ['square_meters', 'num_crimes', 'price'])

# Drop what's missing for everything else
df_train = df_train.dropna()

# Combining dataframes together again for imputation
df = pd.concat([df_train, df_test], axis=0).sort_values("id").reset_index()

# Replacing missing values in price with 0
df['price'] = df['price'].fillna(0)

# Imputing everything (not using price) aside from the binary variables
cols_to_impute = ['num_rooms', 'num_baths', 'square_meters','year_built', 'num_crimes', 'floor', 'neighborhood_crime_encoded']
pred_cols = ['num_rooms', 'num_baths', 'square_meters','year_built', 'num_crimes', 'floor', 'neighborhood_crime_encoded']

df = reg_imputer(df, cols_to_impute, pred_cols)

# Split dataframes again to train model
df_test = df[df['price'] == 0]
df_train = df[df['price'] != 0]


# Train model without binary variables
df_no_binary = df_train[['num_rooms', 'num_baths', 'square_meters', 'year_built', 'floor', 'num_crimes', 'neighborhood_crime_encoded', 'floor_one_dummy', 'num_supermarkets']]

y_train = df_train['price']
x_train = df_no_binary

model_no_binary = LinearRegression()
model_no_binary.fit(x_train, y_train)


# Train model with all variables
y_train = df_train['price']
x_train = df_train[['num_rooms', 'num_baths', 'square_meters', 'year_built', 'floor', 'num_crimes', 'neighborhood_crime_encoded', 'is_furnished', 'has_pool', 'num_crimes', 'has_ac', 'accepts_pets', 'floor_one_dummy', 'num_supermarkets']]

model = LinearRegression()
model.fit(x_train, y_train)



# Subsetting test data for binary variables missing
binary_cols = ['is_furnished', 'has_pool', 'has_ac', 'accepts_pets']
df_missing = df_test[df_test[binary_cols].isna().any(axis=1)]
df_not_missing = df_test[~df_test[binary_cols].isna().any(axis=1)]


# Drop binaries from df missing
df_missing.drop(binary_cols, axis=1, inplace=True)



# Prediction for df_not_missing
x_test = df_not_missing[['num_rooms', 'num_baths', 'square_meters', 'year_built', 'floor', 'num_crimes', 'neighborhood_crime_encoded', 'is_furnished', 'has_pool', 'num_crimes', 'has_ac', 'accepts_pets', 'floor_one_dummy', 'num_supermarkets']]
y_pred_not_missing = model.predict(x_test)

df_not_missing['pred'] = y_pred_not_missing

# Prediction for df_missing
x_test = df_missing[['num_rooms', 'num_baths', 'square_meters', 'year_built', 'floor', 'num_crimes', 'neighborhood_crime_encoded', 'floor_one_dummy', 'num_supermarkets']]
y_pred_missing = model_no_binary.predict(x_test)

df_missing['pred'] = y_pred_missing
new_df = pd.DataFrame()

# Creating final DataFrame
new_df['id'] = df_missing['id'].tolist() + df_not_missing['id'].tolist()
new_df['pred'] = df_missing['pred'].tolist() + df_not_missing['pred'].tolist()

new_df.to_csv('./fixed_floor_old_ollie_file_with_supermarkets.csv', index=False)
