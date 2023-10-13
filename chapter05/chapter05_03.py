# 독립변수가 여럿인 모델
import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import statsmodels.formula.api as smf
import statsmodels.api as sm

sales = pd.read_csv('E://python_study//statistical_simulation//sample//5-3-1-lm-model.csv')
# 데이터 형태 확인하기
print(sales.head(3))
sns.pairplot(data=sales, hue=  'weather', palette='gray')
# 복수의 독립변수를 가진 모델을 추정
lm_sales = smf.ols(
    'sales ~ weather + humidity + temperature + price',
    data=sales).fit()
print(lm_sales.summary())
# type 1 anova분석
print(sm.stats.anova_lm(lm_sales, typ=1).round(3))

# 만약 독립변수의 순서를 바꾼다면 검정 결과가 달라진다.(모델 계수는 같지만 p-value가 달라짐)
lm_sales2 = smf.ols(
    'sales ~ weather + temperature + humidity + price',
    data=sales).fit()
print(lm_sales2.summary())
print(sm.stats.anova_lm(lm_sales2,typ=1).round(3))

# Type 2 ANOVA의 경우 독립변수의 순서가 바뀌게 되어도 검정 결과가 변하지 않는다.
mod_full = smf.ols(
    'sales ~ weather + humidity + temperature + price', sales).fit()
print(mod_full.summary())
print(sm.stats.anova_lm(mod_full,typ=2).round(3))
# 검정 결과 humidity의 p-value가 0.578이므로 humidity의 변수는 제거하는 것이 합리적이다
mod_non_humi = smf.ols(
    'sales ~ weather + temperature + price', sales).fit()
print(mod_non_humi.summary())
print(sm.stats.anova_lm(mod_non_humi, typ=2).round(3))

# AIC를 사용한 선택
print("모든 변수를 포함한 모델: ",mod_full.aic.round(3))
print("습도를 제외한 모델: ", mod_non_humi.aic.round(3))
plt.show()

