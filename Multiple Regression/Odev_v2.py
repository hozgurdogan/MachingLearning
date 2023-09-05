import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

 
veriler=pd.read_csv('odev_tenis.csv')
# ---> verileri check etme print(veriler)

# Satir sayisini hesapla

satir_Sayisi=len(veriler)

# ----> Gerekli kütüphane tanımlamaları 
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing 


#----> katagorik veriyi numeric veriye çevirme işlemleri

ohe=preprocessing.OneHotEncoder()

veriler2=veriler.apply(LabelEncoder().fit_transform)


Outlook=ohe.fit_transform(veriler2.iloc[:,0:1].values).toarray()

WindyPlayNumeric=(veriler2.iloc[:,3:5].values)

#DataFrameleri oluşturmak

Tempature=(veriler.iloc[:,1:2].values)

sonucForOutlook=pd.DataFrame(data=Outlook,index=range(satir_Sayisi),columns=['kapalı','Yağmurlu','Güneşli'])

print(sonucForOutlook)

sonVeriler=pd.concat([sonucForOutlook,veriler.iloc[:,1:3]],axis=1)

print(sonVeriler)


sonucForWindyPlay=pd.DataFrame(data=WindyPlayNumeric,index=range(satir_Sayisi),columns=['Rüzgar','Oynamak'])

print(sonucForWindyPlay)

sonVeriler=pd.concat([sonucForWindyPlay,sonVeriler],axis=1)

#Veriler uygun hale getirildi şimdi train ve test ayrıcaz


from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test=train_test_split(sonVeriler.iloc[:,0:6],sonVeriler.iloc[:,6:7], test_size=0.33,random_state=0)


#Gerekli lineerModeli uygulucaz

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()

regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)



#Backward Eliminatiton uyguluyoruz model doğruluğu için





import statsmodels.api as sm

X=np.append(arr=np.ones((satir_Sayisi,1)).astype(int), values=sonVeriler.iloc[:,:-1],axis=1)

X_l=sonVeriler.iloc[:,[0,1,2,3,4,5]].values
r_ols=sm.OLS(endog=sonVeriler.iloc[:,-1:],exog=X_l)


r=r_ols.fit()


print(r.summary())






sonVeriler=sonVeriler.iloc[:,1:]

X=np.append(arr=np.ones((satir_Sayisi,1)).astype(int), values=sonVeriler.iloc[:,:-1],axis=1)

X_l=sonVeriler.iloc[:,[0,1,2,3,4]].values
r_ols=sm.OLS(endog=sonVeriler.iloc[:,-1:],exog=X_l)


r=r_ols.fit()


print(r.summary())


x_train=x_train.iloc[:,1:]




y_train=y_train.iloc[:,1:]


x_train,x_test,y_train,y_test=train_test_split(sonVeriler.iloc[:,0:5],sonVeriler.iloc[:,5:6], test_size=0.33,random_state=0)

regressor=LinearRegression()

regressor.fit(x_train,y_train)

y_pred2=regressor.predict(x_test)


