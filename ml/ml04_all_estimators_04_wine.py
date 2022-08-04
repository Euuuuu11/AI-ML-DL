# keras15
import numpy as np
from sklearn.datasets import load_wine
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
# 웨이트의 난수

#1. 데이터
datasets = load_wine()
x = datasets.data
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
# AdaBoostClassifier 의 정답률 :  0.8611111111111112
# BaggingClassifier 의 정답률 :  0.9166666666666666
# BernoulliNB 의 정답률 :  0.4444444444444444
# CalibratedClassifierCV 의 정답률 :  0.9166666666666666
# CategoricalNB 은 안나온 놈!!
# ClassifierChain 은 안나온 놈!!
# ComplementNB 의 정답률 :  0.7777777777777778
# DecisionTreeClassifier 의 정답률 :  0.8611111111111112
# DummyClassifier 의 정답률 :  0.4444444444444444
# ExtraTreeClassifier 의 정답률 :  0.9166666666666666
# ExtraTreesClassifier 의 정답률 :  0.9722222222222222
# GaussianNB 의 정답률 :  0.9722222222222222
# GaussianProcessClassifier 의 정답률 :  0.5
# GradientBoostingClassifier 의 정답률 :  0.8611111111111112
# HistGradientBoostingClassifier 의 정답률 :  0.9444444444444444
# KNeighborsClassifier 의 정답률 :  0.8611111111111112
# LabelPropagation 의 정답률 :  0.5
# LabelSpreading 의 정답률 :  0.5
# LinearDiscriminantAnalysis 의 정답률 :  0.9444444444444444
# LinearSVC 의 정답률 :  0.7222222222222222
# LogisticRegression 의 정답률 :  0.9444444444444444
# LogisticRegressionCV 의 정답률 :  0.8888888888888888
# MLPClassifier 의 정답률 :  0.9722222222222222
# MultiOutputClassifier 은 안나온 놈!!
# MultinomialNB 의 정답률 :  0.9722222222222222
# NearestCentroid 의 정답률 :  0.8055555555555556
# NuSVC 의 정답률 :  0.8055555555555556
# OneVsOneClassifier 은 안나온 놈!!
# OneVsRestClassifier 은 안나온 놈!!
# OutputCodeClassifier 은 안나온 놈!!
# PassiveAggressiveClassifier 의 정답률 :  0.7777777777777778
# Perceptron 의 정답률 :  0.5277777777777778
# QuadraticDiscriminantAnalysis 의 정답률 :  0.9722222222222222
# RadiusNeighborsClassifier 은 안나온 놈!!
# RandomForestClassifier 의 정답률 :  0.9444444444444444
# RidgeClassifier 의 정답률 :  0.9722222222222222
# RidgeClassifierCV 의 정답률 :  0.9722222222222222
# SGDClassifier 의 정답률 :  0.7777777777777778
# SVC 의 정답률 :  0.7777777777777778
# StackingClassifier 은 안나온 놈!!
# VotingClassifier 은 안나온 놈!!