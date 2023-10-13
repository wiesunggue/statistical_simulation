# CH5 정규선형모델
# 5.1 연속형 독립변수가 하나인 모델

import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import statsmodels.formula.api as smf
import statsmodels.api as sm

beer = pd.read_csv('E://python_study//statistical_simulation//sample//5-1-1-beer.csv')
print(beer.head())
print(len(beer))
sns.jointplot(x='temperature',y='beer',data=beer, color='black')
# 대체적으로 선형 증가하는 것으로 판단됨 -> 맥주 매상~N(a+bx,sigma)이라는 가정을 해보자

lm_model = smf.ols(formula='beer~temperature',data=beer).fit()
# ols는 최소 제곱법을 사용한 추정
print(lm_model.summary())

# AIC를 사용한 모델 선택
null_model = smf.ols('beer ~ 1', data = beer).fit()
print(null_model.summary())

# null 모델보다 AIC가 더 작으니 lm_model을 선택하는 것이 더 합리적이다
# 회귀 직선 그리기 -> seaborn 활용하면 된다.
sns.lmplot(x='temperature',y='beer',data=beer,
           scatter_kws={'color':'black'},
           line_kws={'color':'black'})

# 모델을 활용한 예측하기
print(lm_model.predict()) # 아무것도 입력하지 않으면 훈련데이터를 사용해서 에측 결과를 반환한다.

# 데이터를 사용한 예측 하기
print(lm_model.predict({"temperature":[i for i in range(10)]}))

# 잔차 계산하기
resid = lm_model.resid
print(resid)

# 결정 계수 -> 모델의 설명력 기준
print(lm_model.rsquared)

# 수정된 결정계수 -> 모델의 설명력 기준 + 변수의 개수가 추가됨에 패널티
print(lm_model.rsquared_adj)

# 잔차는 정규 분포를 따라야 함
sns.histplot(resid,color='black')

# Q-Q Plot 그리기
fig = sm.qqplot(resid,line='s')

# 정규분포의 왜도(3차 적률)=0, 첨도(4차 적률)=3

plt.show()

# Durbin-Watson 지표는 자기상관성에 대한 평가
# 통계량이 2보다 크게 차이나면 문제가 있음 -> 자기 상관이 있으므로 t검정 결과를 신뢰할 수 없음
# 2보다 크게 차이가 난다면 일반화 제곱법 등의 사용을 고려해야 함

