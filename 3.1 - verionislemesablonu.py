import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


veriler=pd.read_csv("satislar.csv")
 

aylar=veriler[["Aylar"]]

satislar=veriler.iloc[:,1:2].values


from sklearn.model_selection import train_test_split


x_train, x_test,y_train,y_test=train_test_split(aylar,satislar,test_size=0.33,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

X_train=sc.fit_transform(x_train)
Y_train=sc.fit_transform(y_train)