import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense
import pandas as pd
from sklearn.svm import LinearSVC # 레거시한 리니어 모델 사용


import tensorflow as tf
tf.random.set_seed(66)
# 웨이트의 난수

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (581012, 54) (581012,)
print(np.unique(y, return_counts=True)) 


x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.8, shuffle=True, random_state=123)

#2. 모델구성
model = LinearSVC()


#3. 컴파일, 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test) # evaluate 대신 score 사용
print('결과 acc :', result)

# LinearSVC 결과
# 결과 acc : 0.5310534151441874

