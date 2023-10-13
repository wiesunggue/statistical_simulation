# 6.3 로지스틱 회귀
# 이항분포를 따르는 확률분포의 회귀분석
# 예제 : 공부시간(x)에 따른 시험의 합격과 불합격의 확률(p) 예측

# 로짓 함수 f(x) = log(x/(1-x))
# 로지스틱 함수 g(y) = 1/(1+exp(-y))

# 즉 풀고자 하는 문제는 log(p/(1-p))=b0+b1*공부시간(x) 이 된다
# 우리가 알고 있는 데이터는 A, B, C... 사람의 공부시간과 합불여부이다

import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import statsmodels.formula.api as smf
import statsmodels.api as sm

test_result = pd.read_csv('E://python_study//statistical_simulation//sample//6-3-1-logistic-regression.csv')
print(test_result.head())

# 그래프를 그리면 공부시간 마다 합격률이 높아진다.
sns.barplot(x='hours',y='result',data=test_result, palette='gray_r')

# 그렇다면 공부시간마다의 합격률은?
print(test_result.groupby('hours').mean())

# 로지스틱 회귀 실습
mod_glm = smf.glm(formula='result~hours', # 모델의 가정 식
                  data=test_result, # 데이터
                  family=sm.families.Binomial()).fit() # glm에서 설정할 분포(이항분포)
print(mod_glm.summary())

# NULL 모델과 비교하기
null_glm = smf.glm(formula='result~1', # 모델의 가정 식
                  data=test_result, # 데이터
                  family=sm.families.Binomial()).fit() # glm에서 설정할 분포(이항분포)
print(null_glm.summary())

# aic가 줄었으니 해당 모델은 유효하다
print(mod_glm.aic, null_glm.aic)

# 로지스틱 회귀곡선 그래프
sns.lmplot(x='hours',y='result',
           data= test_result,
           logistic=True,
           scatter_kws={'color':'black'},
           line_kws={'color':'black'},
           x_jitter = 0.1, y_jitter= 0.02)

# 공부시간에 따른 성공확률 구하기
exp_val = pd.DataFrame({'hours': np.arange(0,10,1)})
pred = mod_glm.predict(exp_val)
print(pred)

# 오즈와 오즈비
# 오즈 = 성공확률/실패확률 = (p/1-p)
# 오즈비 = 오즈1/오즈2
# 로즈오즈비 = log(오즈비)

# 로지스틱 회귀계수와 오즈비의 관계
# 독립변수를 1시간 증가시킬 때의 로즈오즈비 = 회귀계수
# 회귀계수 = 0.9288인데 만약 pred변수에서 odds1=pred[k]/(1-pred[k]), odds2=(pred[k+1]/(1-pred[k+1])) 이라 하면
# log(odds2/odds1) = 0.9288이 된다
odds = [0]*10
for i in range(10):
    odds[i] = pred[i]/(1-pred[i])
for i in range(9):
    odds_ratio = np.log(odds[i+1]/odds[i])
    print(odds_ratio)
# 항상 odds비는 유지된다

plt.show()