# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 18:50:13 2020

@author: sadievrenseker
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('satislar.csv')
#pd.read_csv("veriler.csv")
#test

aylar=veriler[['Aylar']]
satislar=veriler[['Satislar']]


print(veriler)

print(aylar)

print(satislar)



from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(aylar,satislar,test_size=0.33,random_state=0)


'''
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train =sc.fit_transform(x_train)

X_test=sc.fit_transform(x_test)

Y_train=sc.fit_transform(y_train)

Y_test=sc.fit_transform(y_test)

'''


#model inşası 


from sklearn.linear_model import LinearRegression

lr=LinearRegression()

lr.fit(x_train,y_train)


x_train=x_train.sort_index()
y_train=y_train.sort_index()
tahim=lr.predict(x_test)






plt.plot(x_train,y_train)


plt.plot(x_test,tahim)





