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
from sklearn.model_selection import KFold, cross_val_score

import tensorflow as tf
tf.random.set_seed(66)
# 웨이트의 난수

#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target


x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.8, shuffle=True, random_state=68)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

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
        
        scores = cross_val_score(model, x_test, y_test, cv=5)  
        print(name , scores, '\n cross_val_score : ', round(np.mean(scores), 4))
    except :
        # continue    
        print(name, '은 안나온 놈!!')  
   
   
   
        
# 모델의 개수 :  41
# AdaBoostClassifier [0.75       0.85714286 1.         0.71428571 0.71428571] 
#  cross_val_score :  0.8071
# BaggingClassifier [1.         0.85714286 1.         0.85714286 0.71428571] 
#  cross_val_score :  0.8857
# BernoulliNB [0.5        0.42857143 0.42857143 0.42857143 0.42857143] 
#  cross_val_score :  0.4429
# CalibratedClassifierCV [0.875      1.         0.85714286 0.85714286 0.85714286] 
#  cross_val_score :  0.8893
# CategoricalNB [0.875   nan   nan   nan   nan] 
#  cross_val_score :  nan
# ClassifierChain 은 안나온 놈!!
# ComplementNB [0.75       0.85714286 0.85714286 0.71428571 0.57142857]
#  cross_val_score :  0.75
# DecisionTreeClassifier [0.875      1.         0.85714286 1.         0.85714286] 
#  cross_val_score :  0.9179
# DummyClassifier [0.5        0.42857143 0.42857143 0.42857143 0.42857143]
#  cross_val_score :  0.4429
# ExtraTreeClassifier [0.75       0.57142857 1.         0.71428571 0.85714286]
#  cross_val_score :  0.7786
# ExtraTreesClassifier [1.         1.         1.         0.85714286 0.71428571] 
#  cross_val_score :  0.9143
# GaussianNB [1.         1.         1.         1.         0.85714286]
#  cross_val_score :  0.9714
# GaussianProcessClassifier [0.375      0.42857143 0.14285714 0.42857143 0.28571429] 
#  cross_val_score :  0.3321
# GradientBoostingClassifier [1.         0.85714286 0.71428571 1.         0.85714286] 
#  cross_val_score :  0.8857
# HistGradientBoostingClassifier [0.5        0.42857143 0.42857143 0.42857143 0.42857143] 
#  cross_val_score :  0.4429
# KNeighborsClassifier [0.5        1.         0.71428571 0.71428571 0.71428571]
#  cross_val_score :  0.7286
# LabelPropagation [0.375      0.71428571 0.42857143 0.42857143 0.28571429] 
#  cross_val_score :  0.4464
# LabelSpreading [0.375      0.71428571 0.42857143 0.42857143 0.28571429]
#  cross_val_score :  0.4464
# LinearDiscriminantAnalysis [1.         1.         1.         0.85714286 0.85714286] 
#  cross_val_score :  0.9429
# LinearSVC [0.75       1.         0.71428571 0.71428571 0.42857143] 
#  cross_val_score :  0.7214
# LogisticRegression [1. 1. 1. 1. 1.] 
#  cross_val_score :  1.0
# LogisticRegressionCV [1.         1.         1.         1.         0.85714286] 
#  cross_val_score :  0.9714
# MLPClassifier [1.         0.42857143 0.42857143 0.42857143 1.        ] 
#  cross_val_score :  0.6571
# MultiOutputClassifier 은 안나온 놈!!
# MultinomialNB [0.75       1.         1.         0.85714286 0.85714286]
#  cross_val_score :  0.8929
# NearestCentroid [0.75       0.85714286 0.71428571 0.71428571 0.71428571]
#  cross_val_score :  0.75
# NuSVC [0.625      1.         0.71428571 0.71428571 0.85714286] 
#  cross_val_score :  0.7821
# OneVsOneClassifier 은 안나온 놈!!
# OneVsRestClassifier 은 안나온 놈!!
# OutputCodeClassifier 은 안나온 놈!!
# PassiveAggressiveClassifier [0.5        0.71428571 0.71428571 0.71428571 0.57142857] 
#  cross_val_score :  0.6429
# Perceptron [0.25       0.85714286 0.42857143 0.57142857 0.71428571]
#  cross_val_score :  0.5643
# QuadraticDiscriminantAnalysis [0.75       0.42857143 0.42857143 0.42857143 0.42857143] 
#  cross_val_score :  0.4929
# RadiusNeighborsClassifier [nan nan nan nan nan]
#  cross_val_score :  nan
# RandomForestClassifier [1.         1.         1.         0.85714286 0.71428571] 
#  cross_val_score :  0.9143
# RidgeClassifier [1.         1.         1.         0.85714286 0.85714286] 
#  cross_val_score :  0.9429
# RidgeClassifierCV [1.         1.         1.         0.85714286 0.85714286]
#  cross_val_score :  0.9429
# SGDClassifier [0.375      0.57142857 0.71428571 0.71428571 0.57142857] 
#  cross_val_score :  0.5893
# SVC [0.75       0.85714286 0.85714286 0.57142857 0.57142857] 
#  cross_val_score :  0.7214
# StackingClassifier 은 안나온 놈!!
# VotingClassifier 은 안나온 놈!!