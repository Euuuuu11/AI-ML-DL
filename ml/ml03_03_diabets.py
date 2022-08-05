from unittest import result
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense
from sklearn.svm import LinearSVR # 레거시한 리니어 모델 사용

import tensorflow as tf
tf.random.set_seed(66)
# 웨이트의 난수

#1. 데이터
datasets = load_diabetes()
print(datasets.DESCR)
print(datasets.feature_names)
x = datasets['data']
y = datasets.target
print(x)
print(y)
print(x.shape, y.shape) # (150, 4) (150,)

# 원핫인코딩은 모델구성 전 데이터 전처리에서 진행
print("y의 라벨값 : ", np.unique(y)) # y의 라벨값 :  [0 1 2] (총 3개가 있다는 것을 알 수 있음)

# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)
# print(y)
# print(y.shape) #(150, 3)

x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.8, shuffle=True, random_state=68)
#셔플을 잘 해주어야 데이터 분류에 오류가 없음
# print(y_train)
# print(y_test)


#2. 모델구성
from sklearn.svm import LinearSVR, SVR
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression, LinearRegression # LinearRegression 회귀 
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
model = RandomForestRegressor()

#3. 컴파일, 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test) # evaluate 대신 score 사용
print('결과 :', result)

# LinearSVR
# 결과 : 0.9369415995905053

# SVR
# 결과 : 0.9424955665861705

# LinearRegression
# 결과 : 0.9311142723847701

# KNeighborsRegressor
# 결과 : 0.9497584541062802

# DecisionTreeRegressor
# 결과 : 0.9516908212560387

# RandomForestRegressor
# 결과 : 0.9813913043478261