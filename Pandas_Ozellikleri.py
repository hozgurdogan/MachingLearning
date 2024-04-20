#Pandas= Panel Data 

#Veri maniplasyonu ve veri analiz için yazılmış açık kaynak kodlu bir python kütüphanesidir 

# Numpy göre daha üst katmanada yapısı olduğu için numpye göre daha yavaş çalışır.



import pandas as pd



Seri1=pd.Series([1,2,3,54,6,3,1,2,34])


print(Seri1)


#İndexleri ile tutulur



# Tip sorgulama type(seri)



print(type(Seri1))


# Temel yapısal özeliklerine erişmek için 

print(Seri1.axes)

# seri içersinde ki verilerin tipleri hakkında bilgi almak için dtype 

print(Seri1.dtype)


#size bilgileri için "size"


print(Seri1.size)


# boyut için "ndim" komutu 

print(Seri1.ndim)



#Seriye bir vektör olarak erişmek istersek

b=Seri1.values

print(b)



# verini ilk n adet değerini gözlemek istersek head(n) kullanılır



print(Seri1.head(5))


# Serinin sondaki verilerine bakmak istersek tail kullamılır

print(Seri1.tail(3))







####            Index Isımlendirme          ####


yeniSeri=pd.Series([11,22,33,44,55,66],index=[1,3,5,8,9,2])


print(yeniSeri)


yeniSeri=pd.Series([11,22,33,44,55,66],index=["a","b","c","f","t","o"])

print(yeniSeri)


b=pd.concat([Seri1,yeniSeri])

print(b)


print(b)
