
#1. kütüphaneler
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


# verileri import etme

veriler=pd.read_csv('maaslar.csv')
print(veriler)


#data frame dilimleme (Slice)

x=veriler.iloc[:,1:2]

y=veriler.iloc[:,2:3]

# linner reg kullanarak
#doğrusal model oluşturma 
from sklearn.linear_model import LinearRegression

regressionLinner=LinearRegression()

regressionLinner.fit(x.values,y.values)


#plt.plot(x,regressionLinner.predict(x.values),color='blue')  # çizgi grafik






#polynomial regression 
# doğrusal olmayan (nonlinear model)oluşturma 

# 2.dereceden polinom
from sklearn.preprocessing import PolynomialFeatures

Reg_Poly=PolynomialFeatures(degree=2)

x_poly=Reg_Poly.fit_transform(x)


regressionLineer2=LinearRegression()

regressionLineer2.fit(x_poly,y)



# 4. dereceden polinom


Reg_Poly3=PolynomialFeatures(degree=4)

x_poly3=Reg_Poly3.fit_transform(x)


regressionLineer3=LinearRegression()

regressionLineer3.fit(x_poly3,y)

#Görelleştirme


plt.scatter(x,y,color='red')
#1.dereceden polinom grafik gösterimi
plt.plot(x,regressionLinner.predict(x),color='blue')


#2.derceden grafik gösterimi

plt.plot(x,regressionLineer2.predict(x_poly),color='black')


#4.dereceden grafik gösterimi..

plt.plot(x,regressionLineer3.predict(x_poly3),color='green')




#tahminler
'''
print(regressionLinner.predict(11))

print(regressionLinner(6.6))






'''


















