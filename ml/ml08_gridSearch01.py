import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score

import tensorflow as tf
# tf.random.set_seed(66)
# 웨이트의 난수

#1. 데이터
datasets = load_iris()
# print(datasets.DESCR)
# print(datasets.feature_names)
x = datasets['data']
y = datasets.target


x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.8, shuffle=True, random_state=68)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

parameters = [
    {"C":[1, 10, 100, 1000], "kernel":["linear"], "degree":[3,4,5]},    # 12
    {"C":[1, 10, 100], "kernel":["rbf"], "gamma": [0.001, 0.0001]},     # 6
    {"C":[1, 10, 100, 1000], "kernel":["sigmoid"],                      # 24                        
     "gamma":[0.01, 0.001, 0.0001], "degree":[3, 4]}
]                                                                       # 총 42번

#2. 모델구성
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron, LogisticRegression # LogisticRegression 분류 모델 사용
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier # 가지치기 형식으로 결과값 도출, 분류형식
from sklearn.ensemble import RandomForestClassifier # DecisionTreeClassifier가 ensemble 엮여있는게 random으로 

# model = SVC(C=1, kernel='linear', degree=3)
model = GridSearchCV(SVC(),parameters, cv=kfold, verbose=1,             # 42 * 5 = 210
                     refit=True, n_jobs=-1)                             # n_jobs는 cpu 사용 갯수


#3. 컴파일, 훈련
model.fit(x_train, y_train)

print("최적의 매개변수 : ", model.best_estimator_)  # 가장 좋은 추정치
# 최적의 매개변수 :  SVC(C=100, gamma=0.001)
print("최적의 파라미터 : ", model.best_params_)
# 최적의 파라미터 :  {'C': 100, 'gamma': 0.001, 'kernel': 'rbf'}

print("best_score_ : ", model.best_score_)        # 정확도
# best_score_ :  0.9666666666666668
print("model.score : ", model.score(x_test, y_test))
# model.score :  0.9666666666666667


#4. 평가, 예측
y_predict = model.predict(x_test)
print("accuracy_score : ", accuracy_score(y_test, y_predict))
# accuracy_score :  0.9666666666666667

y_pred_best = model.best_estimator_.predict(x_test)
print("최적 튠 ACC : ", accuracy_score(y_test,y_pred_best))
# 최적 튠 ACC :  0.9666666666666667