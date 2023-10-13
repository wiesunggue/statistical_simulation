# 3.10 분할표 검정

import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# 카이스퀘어 검정을 사용
# 카이스퀘어 검정은 기본적으로 기대도수가 5이상이어야 함
# 분할표에서 사용

click_data = pd.read_csv('E://python_study//statistical_simulation//sample//3-10-1-click_data.csv')
print(click_data)

cross = pd.pivot_table(
    data = click_data,
    values = 'freq',
    aggfunc = 'sum',
    index = 'color',
    columns = 'click',
)
print(cross)

print(stats.chi2_contingency(cross,correction=False))

# 카이스퀘어 값으로 p값 계산하기
# 카이스퀘어는 일변량에 대해 계산
p_value = 1-stats.chi2.cdf(x=6.666666,df=1)
print(p_value)