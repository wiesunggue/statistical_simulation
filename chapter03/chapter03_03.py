import numpy as np
import scipy as sp
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

x = np.array([0,1,2,3,4,5,6,7,8,9])
y = np.array([2,3,4,3,5,4,6,7,4,8])

#plt.plot(x,y,color='black')
#plt.title('Title')
#plt.xlabel('X')
#plt.ylabel('Y')
#plt.show()

fish_data = np.array([2,3,3,4,4,4,4,5,5,6])

#sns.displot(fish_data,color='black')
#plt.show()


fish_multi = pd.read_csv('E://python_study//statistical_simulation//sample//3-3-2-fish_multi_2.csv')
print(fish_multi.groupby('species').describe())

length_a = fish_multi.query("species=='A'")['length']
length_b = fish_multi.query("species=='B'")['length']

print(length_a)
'''
sns.displot(length_a,bins=5, color='black',kde=False)
sns.displot(length_b,bins=5, color='gray',kde=False)
plt.show()
'''

# Box Plot 그리기
'''
sns.boxplot(
    x='species',
    y='length',
    data=fish_multi,
    color='gray'
)
plt.show()
'''

# 바이올린 플롯 -> 상자 plot대신 커널밀도추정결과를 활용한 그림
'''
sns.violinplot(
    x='species',
    y='length',
    data=fish_multi,
    color='gray'
)
plt.show()
'''

# 막대그래프 -> 평균과 에러바 기능으로 신뢰구간을 표시
'''
sns.barplot(
    x='species',
    y='length',
    data=fish_multi,
    color='gray')

plt.show()
'''

# 산포도 -> 산포도 + 히스토그램이 같이 있는 그래프
'''
cov_data = pd.read_csv('E://python_study//statistical_simulation//sample//3-2-3-cov.csv')
sns.jointplot(
    x='x',
    y='y',
    data=cov_data,
    color='gray')
plt.show()
'''

# 페어 플롯
iris = sns.load_dataset('iris')
iris.head(n = 3)

print(iris.groupby('species').mean())
sns.pairplot(
    iris,
    hue='species',
    palette='gray'
)

plt.show()