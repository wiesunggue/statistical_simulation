# 3.9 평균값의 차이 검정

import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

paired_test_data = pd.read_csv("E://python_study//statistical_simulation//sample//3-9-1-paired-t-test.csv")
print(paired_test_data)

before = paired_test_data.query(
    'medicine == "before"')["body_temperature"]

after = paired_test_data.query(
    'medicine =="after"')["body_temperature"]
before = np.array(before)
after = np.array(after)

print(before)
print(after)

diff = after - before
print(diff)

# 가설 H0 : 차이가 없다 diff=0 vs H1 : 차이가 있다 diff!=0
# 차이의 평균값이 0과 다른지에 대한 검정
print(stats.ttest_1samp(diff, 0))
# p-value가 매우 작으므로 차이가 없다는 가설을 기각

# 같은 검정인데 두 입력을 주면 결과를 출력해준다
print(stats.ttest_rel(after, before))

# 대응표본 vs 독립표본
# 대응표본은 먹기 전과 후에 대한 검정
# 독립표본은 평균값의 차이에 대한 검정

# 독립표본의 t값 = (x평균-y평균)/root(var_x/m+var_y/n), m은 x의 자유도, n은 y의 자유도
# 계산하기
mean_bef = np.mean(before)
mean_aft = np.mean(after)

sigma_bef = np.var(before,ddof=1)
sigma_aft = np.var(after,ddof=1)

m = len(before)
n = len(after)
t_value = (mean_aft-mean_bef)/np.sqrt(sigma_bef/m+sigma_aft/n)
# P-value의 정의 = P(|T|<t) t분포는 자유도만으로 결정됨 -> 자유도=m+n-2
p_value = stats.t.cdf(-t_value,df=m+n-2)
print(t_value,p_value*2)

# 독립표본 t검정
print(stats.ttest_ind(after, before, equal_var=False))

