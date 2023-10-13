# 정규분포와 응용

import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

simulated_sample = stats.norm.rvs(loc=4, scale=0.8, size=1000000)
print(np.sum(simulated_sample<3))
print(np.sum(simulated_sample<3)/len(simulated_sample)) # 3보다 작은 데이터의 개순
print(stats.norm.cdf(loc=4, scale=0.8, x=3)) # -inf 부터 3까지 적분값과 거의 비슷하다

# 하측확률 : 데이터가 어떤 값 이하가 될 '확률' -> 누적분포를 이용해서 구할 수 있다
# stats메소드로 ppf를 이용해서 구할 수 있음
print(stats.norm.ppf(loc=4, scale=0.8, q=0.105)) # 3보다 작을 확률이 약 10.5%였으므로 q=0.105를 하게 되면 거의 3이 나온다

# t 값 구하기
# t-value = (표본평균-모평균)/표준오차
# t-value = (표본평균 - 모평균)/(표준편차/sqrt(N))

t_value_array = np.zeros(10000)
norm_dist = stats.norm(loc=4,scale=0.8)

for i in range(10000):
    sample = norm_dist.rvs(size=10)
    sample_mean = np.mean(sample)
    sample_std = np.std(sample)
    sample_se = sample_std/np.sqrt(len(sample)) # 표준오차
    t_value_array[i] = (sample_mean-4)/sample_se

# 이 분포는 t분포를 따른다
sns.distplot(t_value_array)
plt.show()

# t의 자유도 = n-1
# t의 분산 = n/(n-2)

# t분포는 모분산을 모르는 사오항에서도 표본평균의 분포에 대해 이야기 할 수 있다.
# t분포는 정규분포보다 조금 더 넓게 퍼진 모양이다.
