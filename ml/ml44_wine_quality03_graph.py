from socket import SIO_KEEPALIVE_VALS
import pandas as pd
import numpy as np

#1. 데이터
path = 'D:/study_data/_data/'
data = pd.read_csv(path + 'winequality-white.csv', index_col=None,
                   header=0, sep=';')  

# print(data.shape)   # (4898, 12)
# print(data.describe())  # 데이터의 내용들 나옴.
# print(data.info())

import matplotlib.pyplot as plt
############# 그래프 그려봐!!!! #############
#1. value_counts -> 쓰지마
#2. groupby 써, count() 써!!!!
# ple.bar로 그린다 (quality 컬럼)
count_data = data.groupby('quality')['quality'].count()  # quality컬럼을 그룹지을 것. // 갯수

print(count_data)

# quality
# 3      20
# 4     163
# 5    1457
# 6    2198
# 7     880
# 8     175
# 9       5

plt.bar(count_data.index, count_data)  # 세로, 가로
plt.show()

'''
# pandas를 numpy로 바꾸는 법
data2 = data.to_numpy()
# data = data.values
# print(type(data))   # <class 'numpy.ndarray'>
# print(data.shape)   # (4898, 12)

x = data2[:, :11]
y = data2[:, 11]
# print(x.shape, y.shape) # (4898, 11) (4898,)

# print(np.unique(y, return_counts=True))
# (array([3., 4., 5., 6., 7., 8., 9.]), array([  20,  163, 1457, 2198,  880,  175,    5], dtype=int64))
# print(data['quality'].value_counts())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=123, shuffle=True,
                                                    train_size=0.8, stratify=y)

#2. 모델
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

#3. 훈련
model.fit(x_train,  y_train)

#4. 평가,예측
y_predict = model.predict(x_test)

from sklearn.metrics import accuracy_score, f1_score    

score = model.score(x_test, y_test)
print('model.score : ', score)
print('acc_score : ', accuracy_score(y_test,y_predict))
print('f1_score(macro) : ', f1_score(y_test,y_predict, average='macro'))
print('f1_score(micro) : ', f1_score(y_test,y_predict, average='micro'))
'''