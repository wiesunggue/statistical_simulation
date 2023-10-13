# 3.8 통계적 가설 검정

import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# 1변량 t검정

# 유의미한 차이의 조건 3가지
# 1. 샘플의 개수가 많다.
# 2. 추출 데이터가 믿을만 하다.
# 3. 평균값의 차이가 크다.

# 귀무가설과 대립가설
# 귀무각설 : 기각의 대상이 되는 첫 번째 가설
# 대립가성 : 귀무가설과 대립되는 가설

# p값 : 귀무가설이 맞다고 가정할 때 얻은 결과보다 더 극단적인 결과가 실제로 관측될 확률 -> p값이 매우 작다면 귀무가설은 틀린 것!
# 유의 수준 : 귀무가설을 기각하는 기준이 되는 값

junk_food = pd.read_csv('E://python_study//statistical_simulation//sample//3-8-1-junk-food-weight.csv')
mu = np.mean(junk_food)
df = len(junk_food)-1
sigma = np.std(junk_food,ddof=1)
se = sigma/np.sqrt(len(junk_food))

# mu=50이다가 귀무가설
hmu = 55
t_value = (mu-hmu)/se
print('t_value',t_value)
alpha = stats.t.cdf(t_value,df=df)
print((1-alpha)*2) # 양측검정의 p-value

# mu=50일 때 p-value 구하기(함수)
print('function p',stats.ttest_1samp(junk_food,55))

# 시뮬레이션에 의한 p계산(p값의 의미)

t_value_arr = np.zeros(50000)
size=20
norm_dist = stats.norm(loc=hmu, scale=sigma) # 평균50, 표준편차=sigma인 정규분포
for i in range(50000):
    sample = norm_dist.rvs(size=size)
    sample_mean = np.mean(sample)
    sample_std = np.std(sample,ddof=1)
    sample_se = sample_std/np.sqrt(size)
    t_value_arr[i] = (sample_mean-hmu)/sample_se

# t_value보다 클 확률 + t_value보다 작을 확률
print((sum(t_value_arr>t_value[0])+sum(t_value_arr<-t_value[0]))/50000)

# p-value는 정리하면 귀무가설과 극단적으로 다를 확률을 구하는 과정
# => p-value가 작다면 귀무가설을 기각한다.

