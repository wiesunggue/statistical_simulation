import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

fish_5 = np.array([2,3,4,5,6])
print(np.random.choice(fish_5, size=3, replace=False)) # 비복원 추출하기

# 복원 추출    replace=True
# 비복원 추출  replace=False

fish_100000 = pd.read_csv('E://python_study//statistical_simulation//sample//3-4-1-fish_length_100000.csv')['length']
print(fish_100000.head()) # 맨 앞의 5개의 데이터만 보여줌

print(len(fish_100000)) # 데이터의 개수
N = len(fish_100000)

# 랜덤 추출하기
sampling = np.random.choice(fish_100000,size=3, replace=False)

print(sp.mean(sampling))

# 데이터의 특성 확인하기
print(np.mean(fish_100000),np.var(fish_100000))
# 평균 4, var=0.64
''''
sns.distplot(fish_100000,kde=False, color='black')
plt.show()
x = np.arange(start=1,stop=7.1, step=0.1)
y = stats.norm.pdf(x=x,loc=4, scale=0.8) # 평균=4 표준편차=0.8인 정규분포

plt.plot(x,y,color='black')
plt.show()

'''

# 정규분포를 따르는 무한 난수 생성하기
sampling_norm = stats.norm.rvs(loc=4, scale=0.8, size=10)

# 베타 분포의 난수 생성
sampling_norm2 = stats.beta.rvs(1,10,size=100000)


sns.displot(sampling_norm2,kde=False, color='black')
plt.show()
