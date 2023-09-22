import numpy as np




'''
Numpy array özelikleri...


1- ndim  --------> boyut sayısı  
2- shape --------> boyut bilgisi 
3- szie  --------> toplam eleman sayısı 
4- dtype --------> array veri tipi

'''

a=np.random.randint(10,size=10)
print(a)


print(a.ndim)


print(a.shape)



print(a.dtype)




# 2 boyutlu bi np dizisi oluşturlım

arrray= np.random.randint(10,size=(3,5))


print(arrray)

print(arrray.ndim)

print(arrray.shape)

print(arrray.dtype)



#Numpy array ile 2 bilinmeyenli denklem çözme  

DenklemKatsayıları=np.array([[5,1],[1,3]])

DenkelmEsitikleri=np.array([12,10 ])




x=np.linalg.solve(DenklemKatsayıları,DenkelmEsitikleri)




# deneme 


array([[0, 1, 2],[3, 4, 5],[6, 7, 8]])













