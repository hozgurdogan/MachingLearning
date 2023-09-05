#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 22:44:20 2023

@author: ozgur
"""

#kütüphaneler
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

# veri yükleme 
veriler=pd.read_csv('veriler.csv')
print(veriler)

boy=veriler[['boy']]
print("boy verileri.....")


print(boy)



print ("boy ve cinsiyet  verierli bi arda....")

boyVeCinsiyet=veriler[['boy', 'cinsiyet']]
print(boyVeCinsiyet)
