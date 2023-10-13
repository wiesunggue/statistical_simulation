# ch3.5 표본 통계량 성질

import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

population = stats.norm(loc=4, scale=0.8)
print(population)

sample_mean_array = np.zeros(10000)

np.random.seed(1)
for i in range(10000):
    sample = population.rvs(size=100)
    sample_mean_array[i]=np.mean(sample)

print(sample_mean_array)
print(np.mean(sample_mean_array))
print(np.std(sample_mean_array))
sns.distplot(sample_mean_array)

plt.show()

# sample 사이즈가 커지면 모평균에 가까워진다
size_array = np.arange(start=10,stop=100000,step=100)
sample_mean_array = np.zeros(len(size_array))
sample_std_array = np.zeros(len(size_array))
for i in range(len(size_array)):
    sample = population.rvs(size_array[i])
    sample_mean_array[i]=np.mean(sample)
    sample_std_array[i]=np.var(sample)

plt.plot(size_array,sample_mean_array,color='black')
plt.show()

# sample 사이즈가 커지면 표준편차는 줄어든다.
plt.plot(size_array,sample_std_array)
plt.show()


