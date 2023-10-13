# 3.7 추정

import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

fish = pd.read_csv('E://python_study//statistical_simulation//sample//3-7-1-fish_length.csv')

mu = np.mean(fish)
sigma = np.var(fish,ddof=1)

print(type(mu))
print(type(sigma))


# 신뢰계수 : 구간 추정의 폭에 대한 신뢰 정도를 확률로 표현한 것
# t분포에 의한 신뢰구간
# (표본평균-모평균)/표본오차
# 표본오차 = sqrt(var/(n-1))/n
df = len(fish)-1
sigma = np.std(fish,ddof=1)
se = sigma/np.sqrt(len(fish))

# t의 신뢰구간 구하기
interval = stats.t.interval(alpha=0.95, df=df, loc=mu, scale=se)
print(interval)

# t분포의 값으로 구간 직접 계산하기
t_975 = stats.t.ppf(q=0.975,df=df)
t_025 = stats.t.ppf(q=0.025,df=df)
print(mu+t_025*se,mu+t_975*se)

# t는 대칭분포이므로 t_975 = -t_025
print(t_025,t_975)

# t는 표준화된 분포이기 때문에 평균과 분산을 곱해서 원래의 분포로 돌려주어야 한다.( z검정과 유사)
# t 값*se+mu => 실제 값

# 신뢰구간의 의미
# 모평균=mu, 분산=sigma라고 할 때, k개씩 뽑는 시행을 N번 반복한다고 하자
# 그렇다면 k번 뽑은 데이터에서 표본평균고 표본분산을 이용해서 신뢰구간을 계산할 수 있는데
# 이 구간이 모평균 안에 들어갈 확률이 신뢰구간에 포함될 확률이다.
# 즉 N번 시행 중 interval안에 mu가 포함 될 확률이다.

