import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd


veriler=pd.read_csv('odev_tenis.csv')

print(veriler)
satir_sayisi = len(veriler)


from sklearn import preprocessing 


TemperatureAndHumidity=veriler.iloc[:,1:3].values

ohe=preprocessing.OneHotEncoder()

outlook=veriler.iloc[:,0:1].values

Outlook=ohe.fit_transform(outlook).toarray()


le=preprocessing.LabelEncoder()

windy=veriler.iloc[:,3:4].values
windy[:,0]=le.fit_transform(windy[:,0])
print(windy)

play=veriler.iloc[:,4:5].values
play[:,0]=le.fit_transform(play[:,0])



sonucForOutlook=pd.DataFrame(data=Outlook,index=range(satir_sayisi),columns=['kapalı','Yağmurlu','Güneşli'])

print(sonucForOutlook)


sonucForWindy=pd.DataFrame(data=windy,index=range(satir_sayisi),columns=['Rüzgar'])

print(sonucForWindy)

sonucForTemperatureAndHumidity=pd.DataFrame(data=TemperatureAndHumidity,index=range(satir_sayisi),columns=['Sicaklık','Nem'])
print(sonucForTemperatureAndHumidity)

SonucKarar=pd.DataFrame(data=play,index=range(satir_sayisi),columns=['Oynanır mı'])
print(SonucKarar)




s=pd.concat([sonucForOutlook,sonucForTemperatureAndHumidity],axis=1)

s1=pd.concat([s,sonucForWindy],axis=1)

s2=pd.concat([s1,SonucKarar],axis=1)

print(s2)


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(s1,play,test_size=0.33,random_state=0)



from sklearn.linear_model import LinearRegression
regressor=LinearRegression()

regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)


# farklı metodlar eklenicek model daha iyi hale iyileştirilcek

import statsmodels.api as sm

X=np.append(arr=np.ones((satir_sayisi,1)).astype(int),values=s1,axis=1)

X_l=s1.iloc[:,[0,1,2,3,4]].values

X_l=np.array(X_l,dtype=float)


model=sm.OLS(play,X_l).fit()

print(model.summary())

