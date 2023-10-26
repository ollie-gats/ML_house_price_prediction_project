#%%
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
from sklearn.linear_model import LogisticRegression


# Filling data path
data_path_train = "C:/Users/vanes/Git_repositories/ML_house_price_prediction_project/data//train.csv"
data_path_test = "C:/Users/vanes/Git_repositories/ML_house_price_prediction_project/data//test.csv"

# reading CSV file
df_train = pd.read_csv(data_path_train)
df_test = pd.read_csv(data_path_test)


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


# Dropping supermarkets number
df.drop('num_supermarkets', axis=1, inplace=True)

# Replacing the outliers with NaN in the number of rooms (justify cutoff value: outliers are very high above 10)
df['num_rooms'] = df['num_rooms'].apply(lambda x: x if x<10 else np.nan)

# Replacing the values of square metres < 40 with NaN (change the cutoff value and see the results)
#df.loc[df['square_meters'] < 0, 'square_meters'] = np.nan

print(df['num_rooms'].value_counts())
mean_sqm_to_room = df.groupby('num_rooms')['square_meters'].mean()
print(mean_sqm_to_room)

# seems odd that the mean size of the apartments is nearly the same for all num_rooms
# creating a new variable: square meters per room and look at the outliers (only if square_meter is positive, closer look at negative values later)
def calculating_sqm_per_room(row):
    if row['square_meters'] > 0:
        return row['square_meters']/ row['num_rooms']
    else:
        row['sqm_per_room'] = pd.NA    
df['sqm_per_room'] = df.apply(calculating_sqm_per_room, axis=1)
sns.boxplot(data=df['sqm_per_room'], palette="Set2")
plt.show()


# outliers detected from boxplot, take a closer look

upper_bound = 80
print(upper_bound)
df[df['sqm_per_room']>upper_bound]['num_rooms'].value_counts()

# nearly all rows that have an outlier as sqm_per_room have only 1 room. This doesn't make sense. Change this data with mean of sqm_per_room of non outliers

median_sqm_per_room = df[df['sqm_per_room'] < upper_bound]['sqm_per_room'].median()
print(median_sqm_per_room)

def changing_num_rooms_in_outliers(row):
    if row['square_meters'] >0 and pd.notna(row['num_rooms']):
        if row['sqm_per_room'] < upper_bound:
            return row['num_rooms']
        else:
            return round(row['square_meters'] / median_sqm_per_room,0)
    else:
        return row['num_rooms']    
    
df['num_rooms'] = df.apply(changing_num_rooms_in_outliers, axis=1)

df[df['sqm_per_room']>upper_bound]['num_rooms'].value_counts()
# dealing with negaitve square meters: 100 values
df_sub = df[df['sqm_per_room']<upper_bound]
df[df['square_meters'].isna() & df['num_rooms'].isna()]

#create possible / logical intervals for each numer ob rooms (for 5 not necessary)
intervals_per_room = []
for i in [1,2,3,4]:
    mean = df_sub[df_sub['num_rooms']==i]['square_meters'].mean()
    std_deviation = np.std(df_sub[df_sub['num_rooms']==i]['square_meters'])
    lower_bound = mean - 1.5*std_deviation
    upper_bound = mean + 1.5*std_deviation
    interval_i = [lower_bound, upper_bound]
    intervals_per_room.append(interval_i)

intervals_per_room


#create a function that checks whether the negative values could be the positive ones
def correcting_negative_sqm(row):
    if row['square_meters'] > 0:
        return row['square_meters']
    for i in range(4):  # Iterate through all rooms
        if row['num_rooms'] == i + 1:
            if intervals_per_room[i][0] <= abs(row['square_meters']) <= intervals_per_room[i][1]:
                return abs(row['square_meters'])
    return np.nan  # If none of the conditions are met, return NaN 

df['square_meters'] = df.apply(correcting_negative_sqm, axis=1)

# Calculate the mean square meters for each 'num_rooms' group
mean_sqm_to_room_mapper = df.groupby('num_rooms')['square_meters'].mean()

# Fill missing 'square_meters' values based on 'num_rooms'
def fill_sqm(row):
    if pd.notna(row['square_meters']):
        return row['square_meters']
    num_rooms = row['num_rooms']
    if num_rooms in mean_sqm_to_room_mapper:
        return mean_sqm_to_room_mapper[num_rooms]
    else:
        return None  
    
df['square_meters'] = df.apply(fill_sqm, axis=1)    


# filling the missing values in num_rooms based on mean_sqm_to_room_mapper
def fill_num_rooms(row):
    if pd.notna(row['num_rooms']):
        return row['num_rooms']
    else:
        if pd.notna(row['square_meters']):
            return round(row['square_meters'] / median_sqm_per_room,0)
        else:
            return row['num_rooms']
    
df['num_rooms'] = df.apply(fill_num_rooms, axis=1)

# still 10 rows that don't contain num_room or square_meter, fill them after standardization with KNN

# first fill all non-binary cells
# Standardization
to_standardize = ['square_meters', 'year_built']

for i in to_standardize:
    df[i] = (df[i] - np.mean(df[i])) / np.std(df[i])

# impute square_meters and num_rooms  with KNN
knn_imputer = KNNImputer(n_neighbors=10)  
df['square_meters'] = knn_imputer.fit_transform(df[['square_meters']])
df['num_rooms'] = knn_imputer.fit_transform(df[['num_rooms']]).astype(int)

#recalculating sqm_per_room after standardizing
df['sqm_per_room'] = df['square_meters'] / df['num_rooms']


# look at distribution of num_baths
df['num_baths'].hist()
# looks quite equally distributed -> fill missing values in num_bath with mean
df['num_baths'] = df['num_baths'].fillna(round(df['num_baths'].mean(),0))

df['baths_per_room'] = df['num_baths'] / df['num_rooms']
df.groupby('num_rooms')['num_baths'].mean()


# Looking at neighborhood and num_crimes
neighb_mean_crime = df.groupby('neighborhood')['num_crimes'].mean()
# significant differences between neighborhood and num_crimes -> use this to fill missing values and encode neighborhood
# function to fill missing values of num_crimes with the mean of the corresponding
def fill_num_crimes(row):
    if pd.notna(row['num_crimes']):
        return row['num_crimes']
    neighborhood = row['neighborhood']
    if neighborhood in neighb_mean_crime:
        return neighb_mean_crime[neighborhood]
    else:
        return None
    
df['num_crimes'] = df.apply(fill_num_crimes, axis=1)

df[df['num_crimes'].isna() & df['neighborhood'].isna()]
# still 6 rows, that don't have neighborhood or num_crimes, all from train_dataset -> drop them

df.dropna(subset=['num_crimes'], inplace=True)

# filling NaNs of neighborhood based on their crime rate
def fill_nearest_neighborhood(row):
    if pd.isna(row['neighborhood']):
        nearest_neighborhood = neighb_mean_crime.index[
        np.abs(neighb_mean_crime.values - row['num_crimes']).argmin()]
        return nearest_neighborhood
    else:
        return row['neighborhood']

df['neighborhood'] = df.apply(fill_nearest_neighborhood, axis=1)

#%%
neigh_year_mean = df.groupby('neighborhood')['year_built'].transform('mean')

# Fill missing values with neighborhood-specific mean values
df['year_built'].fillna(neigh_year_mean, inplace=True)
#%%
# now that all num_crimes are filled, you can encode neighborhood and drop the column
# Encoding neighborhood
neighb_mean_crime = df.groupby('neighborhood')['num_crimes'].mean()
df['neighborhood_crime_encoded'] = df['neighborhood'].map(neighb_mean_crime)

neighb_mean_sqm = df.groupby('neighborhood')['square_meters'].mean()
df['neighborhood_sqm_encoded'] = df['neighborhood'].map(neighb_mean_sqm)
df.drop('neighborhood', axis=1, inplace=True)

# Creating floor variable
df[['floor', 'door_num']] = df['door'].str.split('-', n=1, expand=True)
df['floor'] = df['floor'].str[0]
df["floor"] = pd.to_numeric(df["floor"])

# Dropping door and door_num columns (justify: not influential)
df.drop(['door', 'door_num'], axis=1, inplace=True)

# filling missing values with mean
df['floor'] = df['floor'].fillna(round(df['floor'].mean(),0))


# looking at variable 'orientation'
df['orientation'].value_counts()
df.replace('soxth', 'south', inplace=True)

df['orientation'].value_counts()

#looking at prices regarding the orientation
df.groupby('orientation')['price'].mean()

# appartments that are south have the highest mean price -> create dummy_variable for that
df['south'] = (df['orientation'] == 'south').where(df['orientation'].notna(), np.nan)
df.drop('orientation',axis=1, inplace=True)


#%%
df.isna().sum()
#%%

# 200 missing values for each binary variable. Try: predict the missing values with logistic regression if the others
# beginning with the one, that will probably has the least effect and then adding more variables
correlation_matrix = df.corr(method='pearson')

# Calculate the correlation of all columns with respect to the 'price' column
price_correlation = correlation_matrix['price']

# The 'price' column will also have a correlation of 1 with itself. You can remove it if needed.
price_correlation = price_correlation.drop('price')
plt.figure(figsize=(12, 8))
sns.heatmap(price_correlation.to_frame(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.show         


# order to inpute binary variables: has_pool, is_furnished, south, has_ac, accepts_pets
# imputing has_pool
df['has_pool'] = df['has_pool'].map({True: 1, False: 0})
df_train = df[df['has_pool'].notna()]
df_test = df[df['has_pool'].isna()]
X_train = df_train[['num_rooms', 'num_baths', 'square_meters', 'year_built', 'floor', 'num_crimes', 'neighborhood_crime_encoded','neighborhood_sqm_encoded','sqm_per_room']]
y_train = df_train['has_pool']
X_test = df_test[['num_rooms', 'num_baths', 'square_meters', 'year_built', 'floor', 'num_crimes', 'neighborhood_crime_encoded','neighborhood_sqm_encoded','sqm_per_room']]

# Create an instance of the Logistic Regression model
logreg = LogisticRegression()

# Fit the model to the data
logreg.fit(X_train, y_train)

# Predict the class labels
predicted_labels = logreg.predict(X_test)
df.loc[df['has_pool'].isna(), 'has_pool'] = predicted_labels


# imputing is_furnished
df['is_furnished'] = df['is_furnished'].map({True: 1, False: 0})
df_train = df[df['is_furnished'].notna()]
df_test = df[df['is_furnished'].isna()]
X_train = df_train[['num_rooms', 'num_baths', 'square_meters', 'year_built', 'floor', 'num_crimes', 'neighborhood_crime_encoded','neighborhood_sqm_encoded','sqm_per_room', 'has_pool']]
y_train = df_train['is_furnished']
X_test = df_test[['num_rooms', 'num_baths', 'square_meters', 'year_built', 'floor', 'num_crimes', 'neighborhood_crime_encoded','neighborhood_sqm_encoded','sqm_per_room', 'has_pool']]

# Create an instance of the Logistic Regression model
logreg = LogisticRegression()

# Fit the model to the data
logreg.fit(X_train, y_train)

# Predict the class labels
predicted_labels = logreg.predict(X_test)
df.loc[df['is_furnished'].isna(), 'is_furnished'] = predicted_labels

# imputing south
df['south'] = df['south'].map({True: 1, False: 0})
df_train = df[df['south'].notna()]
df_test = df[df['south'].isna()]
X_train = df_train[['num_rooms', 'num_baths', 'square_meters', 'year_built', 'floor', 'num_crimes', 'neighborhood_crime_encoded','neighborhood_sqm_encoded','sqm_per_room', 'has_pool']]
y_train = df_train['south']
X_test = df_test[['num_rooms', 'num_baths', 'square_meters', 'year_built', 'floor', 'num_crimes', 'neighborhood_crime_encoded','neighborhood_sqm_encoded','sqm_per_room', 'has_pool']]

# Create an instance of the Logistic Regression model
logreg = LogisticRegression()

# Fit the model to the data
logreg.fit(X_train, y_train)

# Predict the class labels
predicted_labels = logreg.predict(X_test)
df.loc[df['south'].isna(), 'south'] = predicted_labels

# imputing has_ac
df['has_ac'] = df['has_ac'].map({True: 1, False: 0})
df_train = df[df['has_ac'].notna()]
df_test = df[df['has_ac'].isna()]
X_train = df_train[['num_rooms', 'num_baths', 'square_meters', 'year_built', 'floor', 'num_crimes', 'neighborhood_crime_encoded','neighborhood_sqm_encoded','sqm_per_room', 'has_pool', 'is_furnished', 'south']]
y_train = df_train['has_ac']
X_test = df_test[['num_rooms', 'num_baths', 'square_meters', 'year_built', 'floor', 'num_crimes', 'neighborhood_crime_encoded','neighborhood_sqm_encoded','sqm_per_room', 'has_pool', 'is_furnished', 'south']]

# Create an instance of the Logistic Regression model
logreg = LogisticRegression()

# Fit the model to the data
logreg.fit(X_train, y_train)

# Predict the class labels
predicted_labels = logreg.predict(X_test)
df.loc[df['has_ac'].isna(), 'has_ac'] = predicted_labels


# imputing accepts_pets
df['accepts_pets'] = df['accepts_pets'].map({True: 1, False: 0})
df_train = df[df['accepts_pets'].notna()]
df_test = df[df['accepts_pets'].isna()]
X_train = df_train[['num_rooms', 'num_baths', 'square_meters', 'year_built', 'floor', 'num_crimes', 'neighborhood_crime_encoded','neighborhood_sqm_encoded','sqm_per_room', 'has_pool', 'is_furnished', 'has_ac', 'south']]
y_train = df_train['accepts_pets']
X_test = df_test[['num_rooms', 'num_baths', 'square_meters', 'year_built', 'floor', 'num_crimes', 'neighborhood_crime_encoded','neighborhood_sqm_encoded','sqm_per_room', 'has_pool', 'is_furnished', 'has_ac', 'south']]

# Create an instance of the Logistic Regression model
logreg = LogisticRegression()

# Fit the model to the data
logreg.fit(X_train, y_train)

# Predict the class labels
predicted_labels = logreg.predict(X_test)
df.loc[df['accepts_pets'].isna(), 'accepts_pets'] = predicted_labels


df.isna().sum()
#######################################################NO MISSING VALUES#####################################
#%%
# looking for multicollinearity and sparsity
# try out with lasso and ridge

# Split dataframes again to train model
df_test = df[df['price'].isna()]
df_train = df[df['price'].notna()]


# set feature and target variables
X_train = df_train[['num_rooms', 'num_baths', 'square_meters', 'year_built', 'floor', 'num_crimes', 'neighborhood_crime_encoded', 'neighborhood_sqm_encoded','is_furnished', 'has_pool', 'has_ac', 'accepts_pets', 'sqm_per_room', 'south']]
y_train = df_train['price']

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

X_test = df_test[['num_rooms', 'num_baths', 'square_meters', 'year_built', 'floor', 'num_crimes', 'neighborhood_crime_encoded','neighborhood_sqm_encoded', 'is_furnished', 'has_pool', 'has_ac', 'accepts_pets', 'sqm_per_room', 'south']]
y_pred = model.predict(X_test)

df_test['pred'] = y_pred

new_df = pd.DataFrame()

# Creating final DataFrame
new_df['id'] =  df_test['id'].tolist()
new_df['pred'] = df_test['pred'].tolist()

#%%

new_df.to_csv('C:/Users/vanes/Desktop/BSE/Term 1/Computational Machine Learning/change a lot one regression.csv', index=False)





# %%
