# 6.4 일반선형모델의 평가
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
mod_glm = smf.glm('result~hours',data=test_result,
                  family= sm.families.Binomial()).fit()

pred = mod_glm.predict()
y = test_result.result

# 피어슨 잔차
peason_resid = (y-pred)/np.sqrt(pred*(1-pred))
print(peason_resid.head())
# 피어슨 잔차의 제곱합은 피어슨 카이스퀘어 통계량이 된다.
print(np.sum(peason_resid**2))
print(np.sum(mod_glm.resid_pearson**2))
print(mod_glm.pearson_chi2)

# deviance
# deviance가 크면 모델이 맞지 않다고 평가할 수 있음
# 편차 = 2[log(likelihood(max)-log(likelihood(glm)]로 계산함
# likelihood(max)는 모든 합격 여부를 완전히 예측 가능할 때의 로드우도
# likelihood(glm)는 로지스틱 회귀 계수에 의한 로그우도

# deviance의 차이를 검정하는 것은 우도비 검정
pred = mod_glm.predict()
y = test_result.result
resid_tmp = 0 - sp.log(sp.stats.binom.pmf(k = y, n = 1, p = pred))
deviance_resid = sp.sqrt(2*resid_tmp)*np.sign(y-pred)

print(deviance_resid.head())

# deviance는 devaiance의 잔차 제곱합을 구하면 된다.
print(np.sum(deviance_resid**2))

# 교차 엔트로피 오차
# 결론적으로 deviance를 최소화 하는 것과 같음

