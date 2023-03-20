import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd


veriler=pd.read_csv('eksikveriler.csv')



print(veriler)


# eksik sayısal verierl icin

from sklearn.impute import SimpleImputer

imputer1=SimpleImputer(missing_values=np.nan,strategy="mean")

Yas=veriler.iloc[:,1:4].values
print(Yas)
imputer1=imputer1.fit(Yas[:,1:4])
Yas[:,1:4]=imputer1.transform(Yas[:,1:4])
print(Yas)

