# 3.11 검정결과해석

import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# p-value가 0.05보다 작다면
# "유의미한 차이가 있다"

# p-value가 0.05보다 크다면
# "유의미한 차이가 있다고 말할 수 없다"(차이가 없는게 아님)

# 자주하기 쉬운 실수
# p값이 작을 수록 차이가 적은 것이 아님
# p값이 0.05보다 크면 차이가 없는 것이 아님
# 1-p값은 대립가설이 올바를 확률이 아님

# 1종 오류 : 귀무가설(H0)이 올바르지만 귀무가설을 기각하는 경우 -> p-value
# 2종 오류 : 귀무가설이 틀렸는데 귀무가설을 채택하는 경우