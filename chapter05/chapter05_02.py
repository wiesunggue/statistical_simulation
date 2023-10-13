# 5.2 분산 분석
# 분산 분석은 평균값의 차이를 검정하는 방법
# 분산 분석은 다중 검정을 사용하는 경우 신뢰도가 극단적으로 하락하게 되는데 3개 검정의 경우 3개 다 p-value 기준 0.05를 만족하더라도
# 1-0.95*0.95*0.95=15%확률로 잘못됨(신뢰 불가능)

# F비 검정
# 귀무가설 : 수준 간의 평균값에 차이가 없다.(p-value<0.05 이면 기각)
# 대립가설 : 수준 간의 평균값에 차이가 있다.

# F비 검정 계산법
# F비 = 효과의 분산 크기/오차의 분산 크기
# 효과는 평균 값의 차이(군간변동), 오차는 바이올린 플롯에서 최댓값-최솟값 느낌(군내변동)

import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import statsmodels.formula.api as smf
import statsmodels.api as sm

weather = ['cloudy','cloudy','rainy','rainy','sunny','sunny']
beer = [6,8,2,4,10,12]
weather_beer = pd.DataFrame({'beer':beer,
                             'weather':weather})

sns.boxplot(x='weather',y='beer',data=weather_beer,color='gray')

# ANOVA 분석하기
# 맥주 매상 ~ N(a+비*b+맑음*c,sigma)
anova_model = smf.ols('beer~weather',data=weather_beer).fit()
print(sm.stats.anova_lm(anova_model,typ=2))
print(anova_model.summary())

# 모델 추정하기
print(anova_model.predict())


plt.show()