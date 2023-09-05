import pandas as pd 
import numpy as np 
veriler=pd.read_csv('eksikveriler.csv')
from sklearn.impute import SimpleImputer
imputer1=SimpleImputer(missing_values=np.nan,strategy='mean')
yas=veriler.iloc[:,1:4].values
#print(yas)

# boş veriler eğitidi ve transform ediledildiiyor 
imputer1=imputer1.fit(yas[:,1:4])
yas[:,1:4]=imputer1.fit_transform(yas[:,1:4])
#print(yas)

# katogerik veriler haline getirilmesi lazım 

ulke=veriler.iloc[:,0:1].values
#print(ulke)
from sklearn import preprocessing
ohe=preprocessing.OneHotEncoder()
ulke=ohe.fit_transform(ulke).toarray()

#print(ulke)

sonuc=pd.DataFrame(data=ulke, index=range(22), columns=['fr','tr','us'])
#print(sonuc)

sonuc2=pd.DataFrame(data=yas, index=range(22),columns=['boy','kilo','yas'])

#print(sonuc2)
cinsiyet=veriler.iloc[:,-1].values

sonuc3=pd.DataFrame(data=cinsiyet, index=range(22),columns=['cinsiyet '])
#print(sonuc3)
s=pd.concat([sonuc,sonuc2],axis=1)
s1=pd.concat([s,sonuc3],axis=1)
print(s1)


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(s,sonuc3,test_size=0.33,random_state=0)

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)

