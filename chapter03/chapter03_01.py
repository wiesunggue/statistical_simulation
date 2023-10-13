# 3.1 파이썬을 활용한 기술통계 : 일변량 데이터

import numpy as np
import scipy as sp
from scipy import stats
fish_data = np.array([2,3,3,4,4,4,4,5,5,6])

# 평균
sp.mean(fish_data)

# 분산
sp.var(fish_data,ddof=0)

# 표본 분산
sp.var(fish_data,ddof=1)

# 사분위수 구하기
print(stats.scoreatpercentile(fish_data,25))
print(stats.scoreatpercentile(fish_data,75))