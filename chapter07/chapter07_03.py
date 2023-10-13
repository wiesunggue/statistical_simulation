# 7.3 파이썬을 이용한 리지 회귀와 라소 회귀

import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import statsmodels.formula.api as smf
import statsmodels.api as sm

from sklearn import linear_model

X = pd.read_csv('E://python_study//statistical_simulation//sample//7-3-1-large-data.csv')
print(X.head())

# 자료의 표준화
X -= np.mean(X,axis=0)
X /= np.std(X,ddof=1, axis=0)

# 종속변수 생성하기
np.random.seed(1)
noise = sp.stats.norm.rvs(loc=0, scale=1, size=X.shape[0])
y = X.X_1*5 + noise

large_data = pd.concat([pd.DataFrame({"y":y}),X],axis=1)
#sns.jointplot(y='y', x='X_1', data=large_data, color='black')

lm_statsmodels = sm.OLS(endog = y, exog = X).fit()
print(lm_statsmodels.params.head())

lm_sklearn = linear_model.LinearRegression()
lm_sklearn.fit(X,y)
# 일반적인 선형 회귀로는 제대로 된 추정이 안됨(X1만 관련이 있지만 X100까지 전부 관련이 있다는 판단이 나옴)
#print(lm_sklearn.coef_)

n_alphas = 50
# log스케일로 -2부터 0.7까지 균등하게 값 50개 생성하기
ridge_alphas = np.logspace(-2, 0.7, n_alphas)
print()
print(ridge_alphas)
ridge_coefs = []

# alpha(제약) 값에 따른 Ridge의 변화 살펴보기
for a in ridge_alphas:
    ridge = linear_model.Ridge(alpha=a, fit_intercept=False)
    ridge.fit(X,y)
    ridge_coefs.append(ridge.coef_)

ridge_coefs = np.array(ridge_coefs)
log_alphas = -np.log10(ridge_alphas)
plt.plot(log_alphas, ridge_coefs, color='black')
plt.text(max(log_alphas)+0.1, np.array(ridge_coefs)[0,0], "X_1")
plt.xlim([min(log_alphas)-0.1, max(log_alphas)+0.3])
plt.title("Ridge")
plt.xlabel('-log10(alpha)')
plt.ylabel('Coefficients')
plt.show()
# 리지 회귀 - 최적의 정규화 강도 결정
# RidgeCV라는 함수를 이용하면 됨 -> 10-fold-CV기법을 활용해서 예측
ridge_best = linear_model.RidgeCV(cv=10, alphas = ridge_alphas, fit_intercept=False)
ridge_best.fit(X,y)
# 계산된 최적의 alpha값
print( np.log10(ridge_best.alpha_))

# 추정된 최고의 계수값
print(ridge_best.coef_)
print(max(ridge_best.coef_[1:]))
# X_1만 4.46의 강도를 가지고 나머지 변수들은 작은 값을 가지게 됨(X_1제외 계수의 max=1.29)
print(X.head())
print(y.head())
print(type(X),type(y))

# 라소 회귀 - 벌칙항의 영향
lasso_alphas, lasso_coefs, _ = linear_model.lasso_path(X,y)
log_alphas = -np.log10(lasso_alphas)
plt.plot(log_alphas, lasso_coefs.T, color='black')
plt.text(max(log_alphas)+0.1, lasso_coefs[0,-1],"X_1")
plt.xlim([min(log_alphas)-0.1, max(log_alphas)+0.3])

plt.title("Lasso")
plt.xlabel('- log10(alpha)')
plt.ylabel('Coefficeints')


# Lasso 추정의 최적의 alpha를 구하기
lasso_best = linear_model.LassoCV(
    cv = 10, alphas = lasso_alphas)
lasso_best.fit(X,y)

# Best인 alpha와 회귀 계수 출력하기
print(lasso_best.alpha_)
print(lasso_best.coef_)

plt.show()