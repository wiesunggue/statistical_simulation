# 3.2 파이썬을 활용한 기술통계 : 다변량 데이터
# 깔끔한 데이터
# 데이터를 정리할 때는 되도록 행이 변수의 의미를 가지지 않도록 구성해야 한다.
# 되도록 교차 분석표 형태가 되어야 한다.

# 다변량 데이터 관리하기
import numpy as np
import scipy as sp
import pandas as pd

# 데이터 읽기
fish_multi = pd.read_csv('E://python_study//statistical_simulation//sample//3-2-1-fish_multi.csv')
print(fish_multi)

# 그룹별 데이터 읽기 -> A, B가 있다면 A, B를 기준으로 판단
group = fish_multi.groupby('species')
print(group.mean())

# pandas의 describe기능을 활용하면 통계적 요약하여 보여준다
# count(개수), mean, std, min, 25%, 50%, 75%, max값 전부 보여줌
print(group.describe())

# 교차분석표 구현하기 -> 데이터를 원하는 형태로 가공할 수 있다.
# 피벗 테이블 활용하기
shoes = pd.read_csv('E://python_study//statistical_simulation//sample//3-2-2-shoes.csv')
print(shoes)
cross = pd.pivot_table(
    data = shoes,
    values = 'sales',
    aggfunc = 'prod',
    index = 'store',
    columns = 'color'
)

print(cross)

# 공분산 계산하기
cov_data = pd.read_csv('E://python_study//statistical_simulation//sample//3-2-3-cov.csv')
mu_x = sp.mean(cov_data['x'])
mu_y = sp.mean(cov_data['y'])
N = len(cov_data)

# covariance 계산하기
print(sum((cov_data['x']-mu_x)*(cov_data['y']-mu_y))/(N-1))

# cov 모듈 이용하기
x=cov_data['x']
y=cov_data['y']
print(sp.cov(x,y,ddof=1))

print(sp.corrcoef(x,y))