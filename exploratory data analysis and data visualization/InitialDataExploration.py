import seaborn as sns 

planets = sns.load_dataset("planets")

print(planets.head())

df = planets.copy()  # Copying the dataset

print(df.head())

print(df.tail())

# Dataset structural properties

print(df.info())

# To get only variable information

print(df.dtypes)

# Data type conversion

import pandas as pd

df.method = pd.Categorical(df.method)
print(df.dtypes)

print(df.head())
