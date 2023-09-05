import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

verilerSatis = pd.read_csv('satislar.csv')
print(verilerSatis)

from sklearn.impute import SimpleImputer

imputer1 = SimpleImputer(missing_values=np.nan, strategy="mean")

Satislar = verilerSatis.iloc[:, 1:2]
imputer1 = imputer1.fit(Satislar)
verilerSatis.iloc[:, 1:2] = imputer1.transform(Satislar)

print(verilerSatis)

aylar=verilerSatis[['Aylar']]

satislar=verilerSatis[['Satislar']]



from sklearn.model_selection import train_test_split

X = verilerSatis.iloc[:, 1:2]
y = verilerSatis.iloc[:, 2]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



print(aylar)

print(satislar)







