import pandas as pd
import numpy as np

veriler=pd.read_csv('eksikveriler.csv')

#print(veriler)
from sklearn.impute import SimpleImputer
imputer1=SimpleImputer(missing_values=np.nan,strategy='mean')
yas=veriler.iloc[:,1:4].values


imputer1=imputer1.fit(yas[:,1:4])
yas[:,1:4]=imputer1.fit_transform(yas[:,1:4])


print(yas)
##########katogerik veiler

ulke = veriler.iloc[:,0:1].values
#print(ulke)

from sklearn import preprocessing

le=preprocessing.LabelEncoder()

ulke[:,0]=le.fit_transform(veriler.iloc[:,0])
print(ulke)

ohe=preprocessing.OneHotEncoder()
ulke=ohe.fit_transform(ulke).toarray()
print(ulke)

sonuc=pd.DataFrame(data=ulke, index=range(22), columns=['fr','tr','us'])
print(sonuc)

sonuc2=pd.DataFrame(data=yas, index=range(22),columns=['boy','kilo','yas'])

print(sonuc2)