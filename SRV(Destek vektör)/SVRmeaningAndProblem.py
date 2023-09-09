'''
TR
Teoirik olarak nedir Destek Vektörü Regresyonu nasıl çalışır....
İki farklı sınıfı Ayırmak için oluşturmaya çalıştığımız doğruların arasında 
max margini oluşturmaya çalışırız(mesafe)
O yüzden bu problemi çözerken ('Support Vector Machines ')ları kullanırız


Amaç: sınıflandırma probleminni çözmek ve ayırca bunu yapmak içinde max 
margini bulmak

Olay regresyona geçildiğinde Bu kavram maximum noktayı alabilen margin aralığı 
olarak işlev görür.

Amaç margin değerini küçük tutup olabildiğince nokta içeri alamak



x->wx+b+c 
x->wx+b
x->wx+b-c


bu doğrular arasında kalan noktaları maximize etmeyi hedefler aynı zamanda da
min 1/2 |w^2| değerini hedeflerken bunu başarması amaçlanır.


# ENG
Theoretical Explanation of How Support Vector Regression Works....

When trying to separate two different classes, we aim to create the maximum margin (distance) between the lines we construct.

Therefore, we use Support Vector Machines to solve this problem.

Goal: To solve the classification problem and, in addition, to find the maximum margin for doing so.

When it comes to regression, this concept serves as the range that can capture the maximum point.

The goal is to keep the margin value small while including as many points as possible.

x->wx+b+c
x->wx+b
x->wx+b-c

It aims to maximize the points between these lines and at the same time, achieve the goal of minimizing the value of 1/2 |w^2|.

'''
