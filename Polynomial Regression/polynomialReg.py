import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


#verileri import etem

veriler=pd.read_csv('maaslar.csv')
print(veriler)

x=veriler.iloc[:,1:2]

print(x)


y=veriler.iloc[:,2:3]
print(y)
# linner reg kullanarak

from sklearn.linear_model import LinearRegression


regressionLinner=LinearRegression()

regressionLinner.fit(x.values,y.values)

plt.scatter(x.values, y.values,color='red') #parçalı görünüm 

#plt.plot(x,regressionLinner.predict(x.values),color='blue')  # çizgi grafik


#polynomial Reg
'''
from sklearn.preprocessing import PolynomialFeatures

Reg_Poly=PolynomialFeatures(degree=2)

x_poly=Reg_Poly.fit_transform(x)
print(x_poly)

regressionLineer2=LinearRegression()

regressionLineer2.fit(x_poly,y)
y_pred=regressionLineer2.predict(x_poly)


plt.plot(x,y_pred,color='blue')

'''





from sklearn.preprocessing import PolynomialFeatures

Reg_Poly=PolynomialFeatures(degree=2)

x_poly=Reg_Poly.fit_transform(x)
print(x_poly)

regressionLineer2=LinearRegression()

regressionLineer2.fit(x_poly,y)
y_pred=regressionLineer2.predict(x_poly)


plt.plot(x,y_pred,color='blue')

print("Lineer reg için 6.6 değeri için sonuç:"+"")
print(regressionLinner.predict([[6.6]]))

print("Lineer reg için 11 değeri için sonuç:"+"")

print(regressionLinner.predict([[11]]))

'''
print("poli reg için 6.6 değer")


print(regressionLineer2.predict(Reg_Poly.fit_transform([[6.6]])))

print("Poli reg için 11 değeri için sonuç:"+"")


print(regressionLineer2.predict(Reg_Poly.fit_transform([[11]])))




'''
