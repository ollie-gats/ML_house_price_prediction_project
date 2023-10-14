import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.impute import KNNImputer
import numpy as np

# Fill your data path here
data_path = "C:/Users/gatla/OneDrive/BSE/Computational_machine_learning/Project_1/train.csv"

df = pd.read_csv(data_path)


# Understanding the data
df.head()
df.describe()
df.shape

###############################
# Dealing with missing values # 
###############################

df.isna().sum()
df.isna().mean()
msno.bar(df)

# Orientation and nu_supermarkets are main issue.

# Chack relationship between missing values
sns.heatmap(df.isna(), cbar=False)
# No obvious pattern between missing values

# num_supermarkets has more than 80% of values missing so removing
# no way of accurately filling orientation but may be useful so leaving for now
df.drop('num_supermarkets', axis=1, inplace=True)


###############
# Initial EDA #
###############

bar_vars = ['num_rooms', 'num_baths', 'orientation', 'neighborhood']
hist_vars = ['square_meters', 'year_built', 'num_crimes', 'price']
count_vars = ['is_furnished', 'has_pool', 'has_ac', 'accepts_pets']
# Other: door

# THESE COULD BE TURNED INTO FUNCTIONS
# Visualising all the variables

for col in bar_vars:
    col_df = df.groupby(col)['id'].count().reset_index()
    plt.bar(col_df[col], col_df['id'])
    plt.title(f'Bar chart for {col}')
    plt.xticks(rotation=45)
    plt.show()

# Clear outliers in num_rooms
# Typo in orientation


for col in hist_vars:
    plt.hist(df[col])
    plt.title(f'Histogram for {col}')
    plt.show()

# Some square meters are negative


for col in count_vars:
    sns.set(style="darkgrid")
    plt.figure()
    sns.countplot(x=col, data=df, palette="Set1")
    plt.title(f'Count plot for {col}')

# All count vars look fine


########################################
# Dealing with issues from initial EDA #
########################################


# Visualise num_rooms outliers with boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, y='num_rooms')
plt.title('num_rooms outliers')

# How many obvs over 20 rooms
len(df[df['num_rooms'] > 10])

# Replacing the outliers with NaN
df['num_rooms'] = df['num_rooms'].apply(lambda x: x if x<10 else np.nan)


# Checking how many have type for orientation
len(df[df['orientation'] == 'soxth'])

# Just 1 so removing
df = df[df['orientation'] != 'soxth']


# Check number of obvs with negative sq meters
len(df[df['square_meters'] <= 0])

# Dropping
df = df[(df['square_meters'] > 0) | (df['square_meters'].isna())]


#################
# Normalisation #
#################

norm_cols = ['num_rooms','num_baths','square_meters','year_built','num_crimes','price']
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



# Imputing with KNN
imputer = KNNImputer(n_neighbors=2)
imputed_data = imputer.fit_transform(df_norm)
imputed_df = pd.DataFrame(imputed_data, columns=df_norm.columns)

imputed_df = imputed_df.add_prefix('norm_')

df.reset_index(drop=True, inplace=True)
imputed_df.reset_index(drop=True, inplace=True)
df = pd.concat([df, imputed_df], axis=1)

# WE NEED TO CHECK THE NORMALISED VALUES ARE THE SAME WHEN DE-NORMALISED
# TIDY THE OUTCOME DATAFRAME, DENORMALISE AND REPLACE THE OLD COLS


######################################
# Dealing with categorical variables #
######################################

# Categorical variables to deal with
# orientation (dummy)
# Neighborhood (dummy/target encoding with price/crimes)

# Using oneshot with orientation
df = pd.concat([df, pd.get_dummies(df['orientation'], prefix='orien')], axis=1)
# NEED TO BE CAREFUL HERE AS DUE TO THE MISSING VALUES SOME OF THESE WILL JUST ALL BE FALSE

# Target encoding neighborhood with house prices
mean_price_by_neighborhood = df.groupby('neighborhood')['price'].mean()

# Creating categorial variable neighborhood_p
df['neighborhood_p'] = df['neighborhood'].map(mean_price_by_neighborhood)
# DISCUS THIS WITH TEAM, IS THIS SENSIBLE OR WOULD JUST ONE HOT ENCODING BE BETTER, MAYBE WE TEST BOTH



###############
# Further EDA #
###############

# Correltaion matrix
numerical_df = df.drop(['id', 'orientation', 'door', 'neighborhood'], axis=1)

numerical_df.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(numerical_df.corr(), annot=True)
plt.show()

# num_crimes and square meters seem to be the most important for predicting price.

# Look at average price per neighborhood
neighborhood_df = df.groupby('neighborhood')['price'].mean().reset_index()
plt.bar(neighborhood_df['neighborhood'], neighborhood_df['price'])
plt.xticks(rotation = 45)







####################
# Linear Modelling #
####################

# Running simple linear model without feature scaling, using  num_crimes and square_meters as predictors
y_train = df['price']
x_train = df[['num_crimes', 'square_meters']]
model = LinearRegression()
model.fit(x_train, y_train)

# Model res
model.coef_
model.intercept_
model.score(x_train, y_train)

# Testing the model
x_test = df_test[['num_crimes', 'square_meters']]

y_pred = model.predict(x_test)

df_pred = pd.DataFrame()
df_pred['id'] = df_test['id']
df_pred['price'] = y_pred

# Prediction has to have 2000 rows, cannot remove data from the test!!! Even those with missing values 
#need tp 
# Exporting prediction
df_pred.to_csv("C:/Users/gatla/OneDrive/BSE/Computational_machine_learning/Project_1/simple_prediction.csv", index=False)




######################
#2nd attempt at running the model
y_train = df['price']
x_train = df['norm_num_rooms']









# Also remember to look at feature scaling.
# Look at how different distributions differ across apartment prices
# Experiment with a specific simple model to see if that's best
# door varibale needs to be dealth with in some way
# Think about all the different iterations of models we want to test, then set
# up some loop which runs through them all and records the ones with the best predictions
# The cleaning steps should be functionalised once we have them finalised


