# %Ing

'''
# Data Cleaning
1- Noisy Data

Example
Persons   Gender   Pregnancy Status
Person 1  Male     1 (YES)
Example Condition is as follows:

2- Missing Data Analysis

3- Outlier Analysis
# Explanation
Outliers are observations in the data that significantly deviate from the general trend or differ greatly from other observations.

They can mislead rule sets or functions created with the principle of generalization!
It leads to bias.

What is an outlier according to whom?
1- Industry knowledge
For example, not modeling houses with an area of 1000 square meters in a house price prediction model.

2- Standard Deviation Approach
The standard deviation of a variable is calculated and added to the mean of the same variable.
1, 2, or 3 times the standard deviation value added above the average is considered as a threshold value, and values above or below this value are defined as outliers.

!!! Threshold Value = Mean + 1 * Standard Deviation
    Threshold Value = Mean + 2 * Standard Deviation !!!

Data above the threshold value can be considered as outliers.

3- Box Plot (Interquartile Range - IQR) Method

After sorting the data from small to large, an outlier definition is made based on the values corresponding to the quartiles (percentiles), i.e., Q1, Q3.

'''


# %Tr
'''
# Veri Temizleme
1- Gürültülü Veri (Noisy Data)

Örnek:
Kişiler   Cinsiyet   Hamilelik Durumu 
Kişi 1    Erkek      1 (EVET)
Başka bir örnek şu şekildedir:

2- Eksik Veri Analizi (Missing Data Analysis)

3- Aykırı Gözlem Analizi (Outlier Analysis)
# Açıklama
Aykırı gözlemler, veride genel eğilimden önemli ölçüde sapma gösteren veya diğer gözlemlerden büyük ölçüde farklı olan gözlemlerdir.

Bunlar, genelleme ilkesiyle oluşturulan kuralları veya işlevleri yanıltabilir!
Önyargıya yol açar.

Aykırı gözlem kimin bakış açısına göre belirlenir?
1- Sektör bilgisi 
Örneğin, bir ev fiyat tahmin modelinde 1000 metrekarelik evleri modellememek.

2- Standart Sapma Yaklaşımı
Bir değişkenin ortalaması hesaplanır ve aynı değişkenin standart sapması eklenir.
1, 2 veya 3 standart sapma değeri ortalamanın üzerine eklenerek bu değer eşik değer olarak kabul edilir ve bu değerden yukarıda veya aşağıda olan değerler aykırı olarak tanımlanır.

!!!     Eşik Değer = Ortalama + 1 * Standart Sapma
        Eşik Değer = Ortalama + 2 * Standart Sapma    !!!
        
Eşik değeri üzerindeki veriler aykırı olarak kabul edilebilir.

3- Kutu Grafiği (Interquartile Range - IQR) Yöntemi

Veriyi küçükten büyüğe sıraladıktan sonra çeyrekliklere (yüzdeliklere), yani Q1, Q3 değerlerine karşılık gelen değerler üzerinden bir eşik değeri hesaplanır ve bu eşik değere göre aykırı değer tanımı yapılır.
'''



