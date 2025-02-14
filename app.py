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

# Convert to DataFrame
data = pd.DataFrame(load_cali.data, columns=load_cali.feature_names)
data['target'] = load_cali.target

# Adjust display settings to show all columns
pd.set_option('display.max_columns', None)

# Display the first few rows
print(data.head())
