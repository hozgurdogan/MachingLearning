# Matplotlib kütüphanesini plt olarak içe aktarır, veri görselleştirmek için kullanılır.
import matplotlib.pyplot as plt 

# Numpy kütüphanesini np olarak içe aktarır, sayısal işlemler için kullanılır.
import numpy as np

# Pandas kütüphanesini pd olarak içe aktarır, veri manipülasyonu ve analizi için kullanılır.
import pandas as pd

# Seaborn kütüphanesini sns olarak içe aktarır, matplotlib tabanlı istatistiksel grafik çizmek için kullanılır.
import seaborn as sns

# Scikit-learn'den RobustScaler'ı içe aktarır, aykırı değerlere karşı dayanıklı ölçeklendirme için kullanılır.
from sklearn.preprocessing import RobustScaler

# Scikit-learn'den LogisticRegression'ı içe aktarır, lojistik regresyon modeli oluşturmak için kullanılır.
from sklearn.linear_model import LogisticRegression

# Scikit-learn'den çeşitli metrik fonksiyonları içe aktarır, modelin performansını ölçmek için kullanılır.
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

# Scikit-learn'den model seçimi araçlarını içe aktarır, veri setini eğitim ve test setlerine ayırmak ve çapraz doğrulama yapmak için kullanılır.
from sklearn.model_selection import train_test_split, cross_validate

from sklearn.preprocessing import RobustScaler

# Numerik sütunların histogramını çizen fonksiyon
def plot_numerical_col(dataframe, numerical_col):
    dataframe[numerical_col].hist(bins=21)
    plt.xlabel(numerical_col)
    plt.show(block=True)

# Hedef değişkene göre özelliklerin ortalamalarını hesaplayan ve yazdıran fonksiyon
def TargetDataAnalyzWithFeatures(df, numerical_col):
    print(df.groupby("Outcome").agg({numerical_col: "mean"}), end="\n\n\n\n\n\n")


def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
    print(variable +"değiştirildi....")
def find_outlier_rows(dataframe, col_name):
        low_limit, up_limit = outlier_thresholds(dataframe, col_name)
        outlier_condition = (dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)
        return dataframe[outlier_condition]



# Veri setini içe aktarma ve df adıyla bir DataFrame oluşturma
df = pd.read_csv('/home/ozgur/Desktop/MuillMachingLearning/DataSet/GeneralDataset/machine_learning-220803-231749 (1)/machine_learning/datasets/diabetes.csv')

# Veri setini yazdırma
print(df)

# Veri setinin boyutunu (satır ve sütun sayısını) yazdırma
df.shape

# Hedef sütunun (Outcome) değerlerinin sayısını yazdırma
print(df["Outcome"].value_counts())
# Hedef sütunun (Outcome) yüzdelik dağılımını hesaplama ve yazdırma
100 * df["Outcome"].value_counts() / len(df)

# Veri setinin istatistiksel özetini yazdırma
df.describe().T

# 'Outcome' hariç tüm sütun adlarını bir listeye atama
cols = [col for col in df.columns if "Outcome" not in col]

# Tüm numerik sütunlar için histogram çizme
for col in cols:
    plot_numerical_col(df, col)

# Hedef değişkene (Outcome) göre her özelliğin ortalamasını hesaplama ve yazdırma
for col in cols:
    TargetDataAnalyzWithFeatures(df, col)

# Veri setindeki eksik değerleri kontrol etme
df.isnull().sum()

# Pandas görüntüleme seçeneklerini ayarlama
pd.set_option('display.max_columns', None)  # Tüm sütunları göster
pd.set_option('display.max_rows', 10)       # İlk 10 satırı göster

# Veri setinin istatistiksel özetini tekrar yazdırma
print(df.describe().T)

# Tüm sütunlar için aykırı değer kontrolü yapma ve yazdırma
for col in cols:
    if check_outlier(df, col):
        print(f"!!!!!!!!1Aykırı değer içeren sütun: {col}")
        replace_with_thresholds(df, col)
        
    else:
        print(f"Aykırı değer içermeyen sütun:{col}")

'''
Her Sayısal Sütun İçin Döngü: for col in cols: ifadesi, 'cols' adlı listede bulunan her bir sayısal sütun için bir döngü başlatır.

RobustScaler Kullanımı: RobustScaler().fit_transform(df[[col]]) ifadesi, RobustScaler'ı kullanarak her bir sayısal sütunu ölçeklemeyi sağlar. RobustScaler, sayısal değerleri medyan ve çeyrekler arası aralık (IQR) kullanarak dönüştürerek aykırı değerlere karşı dayanıklı bir ölçekleme sağlar.

Ölçeklenmiş Değerlerin 'df[col]' Sütununa Atanması: Elde edilen ölçeklenmiş değerler, 'df[col]' sütununa atanır. Yani, orijinal sayısal sütunun değerleri, RobustScaler tarafından dönüştürülmüş hale getirilir ve 'df' DataFrame'inde ilgili sütuna yerleştirilir.

Neden Gerekli?

Model Performansı: Sayısal değerlerin ölçekleri birbirinden farklı olabilir. Ölçekleme, algoritmaların bu farklılıkla başa çıkmasına yardımcı olarak model performansını artırabilir.

Aykırı Değerlere Karşı Direnç: RobustScaler, aykırı değerlere karşı dayanıklı bir ölçekleme sağlar. Bu da, aykırı değerlerin model üzerinde aşırı etkisi olmasını önler.

Optimizasyon Algoritmaları: Bazı optimizasyon algoritmaları (örneğin, Gradient Descent), ölçeklenmiş verilerle daha iyi çalışabilir. Ölçekleme, bu tür algoritmaların daha hızlı ve daha istikrarlı bir şekilde çalışmasına yardımcı olabilir.

Bu nedenlerle, sayısal sütunları ölçeklemek, genellikle makine öğrenimi modelleri için önemli bir veri ön işleme adımıdır.
'''
# Not: RobustScaler, her bir değeri çeyrekliklere göre standartlaştırarak ölçekler,

for col in cols: # Ölçekleme
    df[col]=RobustScaler().fit_transform(df[[col]]) 
  
df.head()
    
  
    
#Model & Prediction

y= df["Outcome"]
X=df.drop(["Outcome"],axis=1)
    
  
log_model= LogisticRegression().fit(X,y)
  

print(log_model.intercept_) # Model denkleminin sabit değeri

print(log_model.coef_) # Model denkeleminin katsayıları

y_pred=log_model.predict(X)

    
  
# İlk 10 elemanı print et
print("Tahmin Edilen (y_pred):", y_pred[0:10])
print("Gerçek Değerler (y):", y[0:10])
