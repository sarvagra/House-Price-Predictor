# %% Importing Libraries
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


# Adjust display settings to show all columns
pd.set_option('display.max_columns', None)

# %% Load Data
# Fetch the California housing dataset
load_cali = fetch_california_housing()
# Split the dataset into training and testing sets
x, xt, y, yt = train_test_split(load_cali.data, load_cali.target, test_size=0.2, random_state=42)

# Convert training data to DataFrame
data = pd.DataFrame(x, columns=load_cali.feature_names)
data['target'] = y  # Add target variable

# Display basic information
print("Training data shape:", x.shape, y.shape)
print("Test data shape:", xt.shape, yt.shape)

# %% Data Overview
print(data.head())  # First 5 rows
print(data.shape)  # Shape of dataset
data.info()  # Column info


# %% Visualization - Pairplot
sns.pairplot(data, height=2.5)
plt.tight_layout()
plt.show()

# %% Visualization - Target Distribution
sns.histplot(data['target'], kde=True)
plt.show()

# %% Calculate Skewness and Kurtosis
print("Skewness:", data['target'].skew())
print("Kurtosis:", data['target'].kurt())

# %% Combining Training and Testing Data
df = pd.DataFrame(np.vstack([x, xt]), columns=load_cali.feature_names)  # Convert arrays to DataFrame
df = shuffle(df, random_state=42).reset_index(drop=True)  # Shuffle and reset index

df['target'] = np.hstack([y, yt])  # Combine target values
print("Integrated data shape:", df.shape)

# %%

df.tail()
df.head()

# %% Statistical Analysis
print(data.describe())  

# %% Data cleaning : handling missing value 
#first visualise null value
plt.figure(figsize=(16,9))
sns.heatmap(df.isnull()) # use cmap="Blues" or "gray"
plt.savefig("Images/")

count_null=df.isnull().sum()
print(count_null)

# %% dropping columns / features, threshold value to drop value 
 
# calculate the null value percent of the col/feature 
null_percent= abs(count_null/df.sum() *100)
print(null_percent)

#find values with >50% null percent 
null_50=null_percent[null_percent>50]
print(null_50)

# null value imputation
# centeral tendency method : for bell shaped graph , use mean. For left/right skewed graph ,  use median. For
# convert data from numerical to catagorical feature...here all features are numerical in california dataset



# %% split data for training and testing
#alredy done : x & y-> training data , xt & yt -> testing data

# feature scaling 
sc=StandardScaler()
sc.fit(x)

# z= x-u/s

p=sc.transform(x)
q=sc.transform(xt)

'''
sc. mean_
sc.n_features_in_
sc.n_samples_seen_
sc. scale_
sc. var_
sc. with_mean
sc. with_ std
'''
# %% Traininfg model 
from sklearn.svm import SVR
from sklearn. linear_model import LinearRegression
from sklearn. linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from xgboost import XGBRegressor

from sklearn.isotonic import IsotonicRegression
# %% creating objects
svr = SVR()
lr = LinearRegression ()
sgdr = SGDRegressor ()
knr = KNeighborsRegressor ()
gpr = GaussianProcessRegressor ()
dtr = DecisionTreeRegressor ()
gbr = GradientBoostingRegressor ()
rfr = RandomForestRegressor ()
xgbr = XGBRegressor ()
mlpr = MLPRegressor ()
ir = IsotonicRegression()

models = {
"a" : ["LinearRegression", lr],
"b" : ["SVR", svr],
"c" : ["SGDRegressor", sgdr],
"d" : ["KNeighborsRegressor", knr],
"e" : ["GaussianProcessRegressor", gpr],
"f" : ["DecisionTreeRegressor", dtr],
"g" : ["GradientBoostingRegressor", gbr],
"h" : ["RandomForestRegressor", rfr],
"i" : ["XBRegressor", xgbr],
"j" : ["MLPRegressor", mlpr],
"k" : ["IsotoncRegression", ir]}

# %%
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer, r2_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Feature scaling
sc = StandardScaler()
x_scaled = sc.fit_transform(x)  # Scaled training data
xt_scaled = sc.transform(xt)    # Scaled test data

# Define models (Removed IsotonicRegression)
models = {
    "a": ["LinearRegression", LinearRegression()],
    "b": ["SVR", SVR()],
    "c": ["SGDRegressor", SGDRegressor()],
    "d": ["KNeighborsRegressor", KNeighborsRegressor()],
    "e": ["GaussianProcessRegressor", GaussianProcessRegressor()],
    "f": ["DecisionTreeRegressor", DecisionTreeRegressor()],
    "g": ["GradientBoostingRegressor", GradientBoostingRegressor()],
    "h": ["RandomForestRegressor", RandomForestRegressor()],
    "i": ["XGBRegressor", XGBRegressor()],
    "j": ["MLPRegressor", MLPRegressor()]
}

# %%
# Function train models
def test_model(model, X_train=x_scaled, y_train=y):
    cv = KFold(n_splits=7, shuffle=True, random_state=45)
    r2 = make_scorer(r2_score)
    r2_val_score = cross_val_score(model, X_train, y_train, cv=cv, scoring=r2)
    return r2_val_score  # Returns an array of scores

# %%
# Train models
models_score = []
for key, (name, model) in models.items():
    print(f"Training model: {name}")
    score = test_model(model, x_scaled, y)
    mean_score = score.mean()  # Compute mean R² score
    print(f"Mean R² Score for {name}: {mean_score:.4f}")
    models_score.append([name, mean_score])

# Convert results to DataFrame for better readability
models_score_df = pd.DataFrame(models_score, columns=["Model", "Mean R² Score"])
print(models_score_df)


# %% hypperparameter tuning
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb

# Define parameter grid
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0]
}

# Perform random search
xgb_model = xgb.XGBRegressor()
random_search = RandomizedSearchCV(xgb_model, param_distributions=param_grid, 
                                   n_iter=10, cv=5, scoring='r2', random_state=42)
random_search.fit(x_scaled, y)

# Print best parameters
print("Best Parameters:", random_search.best_params_)
best_xgb = random_search.best_estimator_


# %% evaluation on unseen data
from sklearn.metrics import r2_score, mean_squared_error

# Predict on test set
y_pred = best_xgb.predict(xt_scaled)

# Evaluate performance
r2 = r2_score(yt, y_pred)
mse = mean_squared_error(yt, y_pred)

print(f"Test R² Score: {r2:.4f}")
print(f"Test MSE: {mse:.4f}")

# %% find the features that contribute the most
import matplotlib.pyplot as plt
importances = best_xgb.feature_importances_

plt.figure(figsize=(10,5))
plt.bar(load_cali.feature_names, importances, color='skyblue')
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature Importance (XGBoost)")
plt.xticks(rotation=45)
plt.show()

# %% save n deploy
import joblib
joblib.dump(best_xgb, "xgboost_model.pkl")

# %%
# Load the trained model
xgbr_loaded = joblib.load("xgboost_model.pkl")

# Predict on test data
y_pred = xgbr_loaded.predict(xt_scaled)  # xt_scaled is your scaled test set

# Display results
print("Predicted values:", y_pred[:10])  # Show first 10 predictions
print("Actual values:", yt[:10])  # Compare with actual values
      
# %%
