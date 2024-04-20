###############################
# Multi Linear Regression
###############################
from builtins import print

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
df=pd.read_csv('/home/ozgur/Desktop/MuillMachingLearning/DataSet/GeneralDataset/machine_learning-220803-231749 (1)/machine_learning/datasets/advertising.csv')


print(df)

X=df.drop('sales',axis=1)

y=df[['sales']]


print(X)


###############################
#Model
###############################


X_train,X_test,y_train,y_test=train_test_split(X,y,
                                               test_size=0.20,random_state=1)


y_train.shape


reg_model=LinearRegression()
reg_model.fit(X_train,y_train)

reg_model.intercept_

#2.90794702 bias

#coefficients (w-weights) [0.0468431 , 0.17854434, 0.00258619]

reg_model.coef_

# Tv:30 v
#Radio:10
# newspaper:40

w0=0.0468431
w1=0.17854434
w2=0.00258619
b=2.90794702
regModel=b+w0* 30  +w1* 10    +w2* 40

print(regModel)




yeniVeri=[[30],[10],[40]]

yeniVeri=pd.DataFrame(yeniVeri).T




reg_model.predict(yeniVeri)






#################################
# TAHMİN BAŞARISI
#################################

# TRAIN RMSE
#Düşük bir RMSE değeri, modelin genellikle gefrçek değerlere yakın tahminler yaptığını gösterir. 
y_pred=reg_model.predict(X_test)
#Test hatası train hatasından daha yüksek çıkar genelde 


print("RMSE :"+str(np.sqrt(mean_squared_error(y_test, y_pred))))



#Train RKARE BAğımsız değişkenlerinbağımlı değişkenleri etkileme açıklama oranıdır
#R-kare, bir modelin bağımsız değişkenlerle bağımlı değişken arasındaki ilişkiyi ,
#ne kadar iyi açıkladığını ölçen bir değerdir
print("RKare"+str(reg_model.score(X_test,y_test)))



np.mean(np.sqrt(-cross_val_score(reg_model,X,y,cv=10,scoring="neg_mean_squared_error")))















