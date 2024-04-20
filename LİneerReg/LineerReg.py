from builtins import print

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.float_format',lambda x: '%.2f' %x)
#Virgülden sonra 2 basamak getir

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

 
# DataSet import
df=pd.read_csv('/home/ozgur/Desktop/MuillMachingLearning/DataSet/GeneralDataset/machine_learning-220803-231749 (1)/machine_learning/datasets/advertising.csv')


print(df.shape) #Satır Stün Şeklinde df bilgisi 



X=df[["TV"]]

y=df[["sales"]]


#Model

reg_model=LinearRegression().fit(X, y)


#sabit(bias - b)
reg_model.intercept_[0]



reg_model.coef_[0][0]


# 150 birimlik tv harcamsı olursa ne kadar satış olması beklenir?

reg_model.intercept_[0]+ reg_model.coef_[0][0]*150

# 500 birimlik tv harcamsı olursa ne kadar satış olması beklenir?

reg_model.intercept_[0]+ reg_model.coef_[0][0]*500


df.describe().T




#Modelin Görselleştirilmesi
g=sns.regplot(x=X, y=y, scatter_kws={'color':'g', 's':9},
              ci=False,color="r")

g.set_title(f"Model Denklemi :Sales= {round(reg_model.intercept_[0],2)}+ Tv*{round( reg_model.coef_[0][0],2)}")
g.set_xlabel("Tv Harcamaları ")
g.set_ylabel("Satış Sayisi")

#x Ekseni -10 dan 310 a akadar sınırlandırdık
plt.xlim(-10,310)
# Y eksewni 0 dan başlasın
plt.ylim(bottom=0)


plt.show()



##################################
# Tahmin Başarısı
##################################



#MSE

y_pred=reg_model.predict(X)

print(y_pred)


mean_squared_error(y, y_pred)

y.mean()

y.std()


#RMSE

print("Rmse:"+str(np.sqrt(mean_squared_error(y, y_pred))))



#MAE

mean_absolute_error(y,y_pred)

# R- Kare

print("Rkare:"+str(reg_model.score(X,y)))
