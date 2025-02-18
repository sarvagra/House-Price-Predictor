import numpy as np
import pandas as pd

import plotly.express as px 
import plotly.graph_objects as go
import plotly.io as pio
pio.templates

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing

# Fetch the California housing dataset
load_cali = fetch_california_housing()

x=load_cali.data
# Convert to DataFrame
data = pd.DataFrame(x, columns=load_cali.feature_names)
data['target'] = load_cali.target

# Adjust display settings to show all columns
pd.set_option('display.max_columns', None)

# Display the first few rows
#print(data.head())
print(data.shape) #tells the number of rows and columns (r,c)
data.info() #get the info of data like dtypes, null or not null , etc

print(data.describe()) #tells about mean, percentiles, etc

#sns.pairplot(data, height=2.5)
#plt.tight_layout()

sns.distplot(data['target']);
plt.show()
print("skewness: %f" % data['target'].skew())
print("kurtosis: %f" % data['target'].kurt())
