from unittest import result
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense


import tensorflow as tf
# tf.random.set_seed(66)
# 웨이트의 난수


#1. 데이터
datasets = load_iris()
print(datasets.DESCR)
print(datasets.feature_names)
x = datasets['data']
y = datasets.target


# x_train, x_test, y_train, y_test = train_test_split(x, y,
#         train_size=0.8, shuffle=True, random_state=68)
n_splits = 5
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=66)

#2. 모델구성
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron, LogisticRegression # LogisticRegression 분류 모델 사용
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier # 가지치기 형식으로 결과값 도출, 분류형식
from sklearn.ensemble import RandomForestClassifier # DecisionTreeClassifier가 ensemble 엮여있는게 random으로 


model = SVC()

#3.4. 컴파일, 훈련, 평가, 예측
# scores = cross_val_score(model, x, y, cv=kfold)
scores = cross_val_score(model, x, y, cv=5)         # 둘다 가능
 
print('ACC : ', scores, '\n cross_val_score : ', round(np.mean(scores), 4))



#  cross_val_score :  0.9667