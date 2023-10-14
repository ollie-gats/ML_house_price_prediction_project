import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
from sklearn.linear_model import LinearRegression

df = pd.read_csv("C:/Users/gatla/OneDrive/BSE/Computational_machine_learning/Project_1/train.csv")


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

# Removing rows with NaN from other columns as there is little for every col
df.dropna(subset = ['num_baths', 'square_meters', 'year_built', 'door', 'is_furnished',
                    'has_pool', 'neighborhood', 'num_crimes', 'has_ac', 'accepts_pets'], inplace=True)

# THIS SHOULD BE CONSIDERED MORE CAREFULLY, LEADS TO APPROX 25,000 OBVS BEING DROPPED


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
len(df[df['num_rooms'] > 20]) # 36

# Removing those 36 obvs
df = df[df['num_rooms'] < 20]

# Checking max room number
df['num_rooms'].max() # Max is now 4


# Checking how many have type for orientation
len(df[df['orientation'] == 'soxth'])

# Just 1 so removing
df = df[df['orientation'] != 'soxth']


# Check number of obvs with negative sq meters
len(df[df['square_meters'] < 0])

# Only 59 so dropping
df = df[df['square_meters'] > 0]





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





# ADJUST THIS SECTION
########################################
# Applying cleaning steps to test data #
########################################
df_test = pd.read_csv("C:/Users/gatla/OneDrive/BSE/Computational_machine_learning/Project_1/test.csv")
# REALLY THIS SHOULD BE DONE SEPERATELY, AS THE TEST DATA MAY HAVE SLIGHTLY DIFFERENT FEATURES
# CLEAN IT LIKE IT'S A NEW DATASET!!!

df_test.drop('num_supermarkets', axis=1, inplace=True)

# Removing rows with NaN from other columns as there is little for every col
df.dropna(subset = ['num_baths', 'square_meters', 'year_built', 'door', 'is_furnished',
                    'has_pool', 'neighborhood', 'num_crimes', 'has_ac', 'accepts_pets'], inplace=True)

df = df[df['num_rooms'] < 20]

df = pd.concat([df, pd.get_dummies(df['orientation'], prefix='orien')], axis=1)

# Target encoding neighborhood with house prices
mean_price_by_neighborhood = df.groupby('neighborhood')['price'].mean()

# Creating categorial variable neighborhood_p
df['neighborhood_p'] = df['neighborhood'].map(mean_price_by_neighborhood)

df_test.dropna(inplace=True)


# Filling for simple regresison below
df_test['square_meters'].fillna(df_test['square_meters'].mean(), inplace=True)
df_test['num_crimes'].fillna(0, inplace=True)





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


# Also remember to look at feature scaling.
# Look at how different distributions differ across apartment prices
# Experiment with a specific simple model to see if that's best
# door varibale needs to be dealth with in some way
# Think about all the different iterations of models we want to test, then set
# up some loop which runs through them all and records the ones with the best predictions
# The cleaning steps should be functionalised once we have them finalised


