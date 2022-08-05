from unittest import result
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense
from sklearn.svm import LinearSVC # 레거시한 리니어 모델 사용
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
tf.random.set_seed(66)


#1. 데이터
datasets = load_breast_cancer()
print(datasets.DESCR)
print(datasets.feature_names)
x = datasets['data']
y = datasets.target


x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.8, shuffle=True, random_state=68)

#2. 모델구성
allAlogrithms = all_estimators(type_filter='classifier')
# allAlogrithms = all_estimators(type_filter='regressor')

# [('AdaBoostClassifier', <class 'sklearn.ensemble._weight_boosting.AdaBoostClassifier'>

print('allAlogrithms : ', allAlogrithms)    # 딕셔너리들이 list 형태로 묶여져있다.
print('모델의 개수 : ', len(allAlogrithms))  # 모델의 개수 :  41

# [예외처리] 에러가 떳을 때 무시하고, 넘어가겠다. 
for (name, algorithm) in allAlogrithms:
    try :
        model = algorithm()
        model.fit(x_train, y_train)
        
        y_predict = model.predict(x_test)
        acc = accuracy_score(y_test, y_predict)
        print(name, '의 정답률 : ', acc)
    except :
        # continue    
        print(name, '은 안나온 놈!!')    


# 모델의 개수 :  41
# AdaBoostClassifier 의 정답률 :  0.9666666666666667
# BaggingClassifier 의 정답률 :  0.9666666666666667
# BernoulliNB 의 정답률 :  0.3
# CalibratedClassifierCV 의 정답률 :  0.9666666666666667
# CategoricalNB 의 정답률 :  0.9
# ClassifierChain 은 안나온 놈!!
# ComplementNB 의 정답률 :  0.7
# DecisionTreeClassifier 의 정답률 :  1.0
# DummyClassifier 의 정답률 :  0.3
# ExtraTreeClassifier 의 정답률 :  0.9666666666666667
# ExtraTreesClassifier 의 정답률 :  0.9666666666666667
# GaussianNB 의 정답률 :  0.9666666666666667
# GaussianProcessClassifier 의 정답률 :  0.9666666666666667
# GradientBoostingClassifier 의 정답률 :  0.9666666666666667
# HistGradientBoostingClassifier 의 정답률 :  0.9666666666666667
# KNeighborsClassifier 의 정답률 :  0.9666666666666667
# LabelPropagation 의 정답률 :  0.9666666666666667
# LabelSpreading 의 정답률 :  0.9666666666666667
# LinearDiscriminantAnalysis 의 정답률 :  0.9666666666666667
# LinearSVC 의 정답률 :  0.9666666666666667
# LogisticRegression 의 정답률 :  0.9666666666666667
# LogisticRegressionCV 의 정답률 :  0.9666666666666667
# MLPClassifier 의 정답률 :  0.9666666666666667
# MultiOutputClassifier 은 안나온 놈!!
# MultinomialNB 의 정답률 :  0.8
# NearestCentroid 의 정답률 :  0.9333333333333333
# NuSVC 의 정답률 :  0.9
# OneVsOneClassifier 은 안나온 놈!!
# OneVsRestClassifier 은 안나온 놈!!
# OutputCodeClassifier 은 안나온 놈!!
# PassiveAggressiveClassifier 의 정답률 :  0.7
# Perceptron 의 정답률 :  0.7
# QuadraticDiscriminantAnalysis 의 정답률 :  0.9666666666666667
# RadiusNeighborsClassifier 의 정답률 :  0.9
# RandomForestClassifier 의 정답률 :  0.9666666666666667
# RidgeClassifier 의 정답률 :  0.9
# RidgeClassifierCV 의 정답률 :  0.9
# SGDClassifier 의 정답률 :  0.7
# SVC 의 정답률 :  0.9
# StackingClassifier 은 안나온 놈!!
# VotingClassifier 은 안나온 놈!!