# 포아송 회귀
# 확률 분포에 포아송 분포를 사용, 링크함수로 로그함수인 GLM모델

# 포아송 회귀 구조
# log[맥주 판매 개수] = b0 + b1*기온
# 맥주 판매 개수 = 포아송 분포의 lambda

import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import statsmodels.formula.api as smf
import statsmodels.api as sm

beer = pd.read_csv('E://python_study//statistical_simulation//sample//6-5-1-poisson-regression.csv')
print(beer.head())

mod_pois = smf.glm('beer_number~temperature', beer, family=sm.families.Poisson()).fit()
print(mod_pois.summary())

mod_pois_null = smf.glm('beer_number~1',beer, family=sm.families.Poisson()).fit()
print(mod_pois_null.summary())

x_plot = np.arange(0,37)
pred = mod_pois.predict(pd.DataFrame({"temperature":x_plot}))

sns.lmplot(y='beer_number',x='temperature',
           data=beer, fit_reg=False,
           scatter_kws={'color':"black"})

plt.plot(x_plot, pred, color='black')
plt.show()

# 회귀계수의 해석
# temperature의 계수가 0.0761인데 여기서 exp를 취하면 1.079가 된다.
# 1.079가 의미하는 것은 온도가 1도 증가할 때 판매량이 몇 배 증가하는지에 대한 비율이다.
print(np.exp(mod_pois.params['temperature']))

