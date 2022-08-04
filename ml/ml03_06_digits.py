import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense
from sklearn.svm import LinearSVC # 레거시한 리니어 모델 사용


import tensorflow as tf
tf.random.set_seed(66)
# 웨이트의 난수

#1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.8, shuffle=True, random_state=68)

#2. 모델구성
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier # 가지치기 형식으로 결과값 도출, 분류형식
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()


#3. 컴파일, 훈련
model.fit(x_train, y_train)


#4. 평가, 예측
result = model.score(x_test, y_test) # evaluate 대신 score 사용
print('결과 acc :', result)

# LinearSVC 결과
# 결과 acc : 0.95

# SVC
# 결과 acc : 0.9888888888888889

# KNeighborsClassifier
# 결과 acc : 0.9805555555555555

# DecisionTreeClassifier
# 결과 acc : 0.8583333333333333

# RandomForestClassifier
# 결과 acc : 0.9777777777777777

# import sklearn as sk
# print(sk.__version__)  0.24.2