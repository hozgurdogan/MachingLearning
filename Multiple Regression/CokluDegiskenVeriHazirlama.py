import pandas as pd
import numpy as np

veriler=pd.read_csv('veriler.csv')

#print(veriler)

Yas=veriler.iloc[:,1:4]
ulke = veriler.iloc[:,0:1].values
cinsiyet=veriler.iloc[:,4:5].values
print(cinsiyet)
#print(ulke)


#encoder :Katagorik ->numaricVeri

from sklearn import preprocessing
'''
le=preprocessing.LabelEncoder()

le=preprocessing.LabelEncoder()

ulke[:,0]=le.fit_transform(veriler.iloc[:,0])
print(ulke)

cinsiyet[:,0]=le.fit_transform(veriler.iloc[:,4])

print(cinsiyet)


'''

ohe=preprocessing.OneHotEncoder()
ulke=ohe.fit_transform(ulke).toarray()
c=ohe.fit_transform(cinsiyet).toarray()

print("Cinsiyet---->")
print(c)


#numpy dizilerini dataFrame donusumu

sonuc=pd.DataFrame(data=ulke, index=range(22), columns=['fr','tr','us'])
print(sonuc)

sonuc2=pd.DataFrame(data=Yas, index=range(22),columns=['boy','kilo','yas'])

print(sonuc2)


sonuc3=pd.DataFrame(data=c[:,:1], index=range(22),columns=['cinsiyet(0:erkek&1:kız)'])
print(sonuc3)
s=pd.concat([sonuc,sonuc2],axis=1)
s1=pd.concat([s,sonuc3],axis=1)
print(s1)



#verilerin eğitim ve test olarak bölünmesi...
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(s,sonuc3, test_size=0.33,random_state=0)





#from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

regressor=  LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)


boy=s1.iloc[:,3:4].values
print(boy)

sol=s1.iloc[:,:3]

sag=s1.iloc[:,4:]

veri=pd.concat([sol,sag],axis=1)



x_train,x_test,y_train,y_test=train_test_split(veri,boy, test_size=0.33,random_state=0)


r2=  LinearRegression()
r2.fit(x_train,y_train)

y_pred=r2.predict(x_test)

