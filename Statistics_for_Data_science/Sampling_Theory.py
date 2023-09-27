# -*- coding: utf-8 -*-
import numpy as np 

population = np.random.randint(0, 80, 10000)

print(population[0:10])

# Sampling
np.random.seed(115)  # Setting a seed for reproducibility

sample = np.random.choice(a=population, size=100)
print(sample[:10])

print("Population mean ----> " + str(population.mean()))  # Mean of the entire population

print("Mean of 100 samples ----> " + str(sample.mean()))  # Mean of 100 samples

sample1 = np.random.choice(a=population, size=100)
sample2 = np.random.choice(a=population, size=100)
sample3 = np.random.choice(a=population, size=100)
sample4 = np.random.choice(a=population, size=100)
sample5 = np.random.choice(a=population, size=100)
sample6 = np.random.choice(a=population, size=100)
sample7 = np.random.choice(a=population, size=100)
sample8 = np.random.choice(a=population, size=100)

mean_calculation = (sample.mean() + sample1.mean() + sample2.mean() + sample3.mean() + sample4.mean() + sample5.mean() + sample6.mean()) / 7

print("Mean of smaller samples ---> " + str(mean_calculation))
