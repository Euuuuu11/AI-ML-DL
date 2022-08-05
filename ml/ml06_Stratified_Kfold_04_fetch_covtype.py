import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.svm import LinearSVC # 레거시한 리니어 모델 사용


import tensorflow as tf
# tf.random.set_seed(66)
# 웨이트의 난수

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.8, shuffle=True, random_state=68)

n_splits = 5
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=66)

#2. 모델구성
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron, LogisticRegression # LogisticRegression 분류 모델 사용
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier # 가지치기 형식으로 결과값 도출, 분류형식
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()        # 모델을 SVC에서 RandomForestClassifier 바꾸니 속도가 빨라졌다.

import time
#3.4. 컴파일, 훈련, 평가, 예측
# scores = cross_val_score(model, x, y, cv=kfold)
start_time = time.time()
scores = cross_val_score(model, x_train, y_train, cv=5)         # 둘다 가능
 
print('ACC : ', scores, '\n cross_val_score : ', round(np.mean(scores), 4))

y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)

end_time = time.time()

print(y_predict)


print(y_test)
acc = accuracy_score(y_test, y_predict)
print('cross_val_predict ACC : ', acc)

print('끝난 시간 : ',end_time - start_time)

# cross_val_predict ACC :  0.9089954648330938
# 끝난 시간 :  431.3595139980316