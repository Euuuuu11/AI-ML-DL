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
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.model_selection import KFold, cross_val_score

import tensorflow as tf
# tf.random.set_seed(66)


#1. 데이터
datasets = load_breast_cancer()
print(datasets.DESCR)
print(datasets.feature_names)
x = datasets['data']
y = datasets.target


# x_train, x_test, y_train, y_test = train_test_split(x, y,
#         train_size=0.8, shuffle=True, random_state=68)

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
        scores = cross_val_score(model, x, y, cv=5)  
        print(name , scores, '\n cross_val_score : ', round(np.mean(scores), 4))
        
    except :
        # continue    
        print(name, '은 안나온 놈!!')    


# 모델의 개수 :  41
# AdaBoostClassifier [0.96666667 0.93333333 0.9        0.93333333 1.        ] 
#  cross_val_score :  0.9467
# BaggingClassifier [0.96666667 0.96666667 0.9        0.96666667 1.        ] 
#  cross_val_score :  0.96
# BernoulliNB [0.33333333 0.33333333 0.33333333 0.33333333 0.33333333] 
#  cross_val_score :  0.3333
# CalibratedClassifierCV [0.93333333 0.9        0.9        0.83333333 1.        ] 
#  cross_val_score :  0.9133
# CategoricalNB [0.9        1.         0.83333333 0.93333333 0.96666667]
#  cross_val_score :  0.9267
# ClassifierChain 은 안나온 놈!!
# ComplementNB [0.66666667 0.66666667 0.66666667 0.66666667 0.66666667]
#  cross_val_score :  0.6667
# DecisionTreeClassifier [0.96666667 0.96666667 0.9        1.         1.        ] 
#  cross_val_score :  0.9667
# DummyClassifier [0.33333333 0.33333333 0.33333333 0.33333333 0.33333333]
#  cross_val_score :  0.3333
# ExtraTreeClassifier [0.9        0.96666667 0.83333333 0.9        1.        ]
#  cross_val_score :  0.92
# ExtraTreesClassifier [0.96666667 0.96666667 0.9        0.93333333 1.        ] 
#  cross_val_score :  0.9533
# GaussianNB [0.93333333 0.96666667 0.93333333 0.93333333 1.        ]
#  cross_val_score :  0.9533
# GaussianProcessClassifier [0.96666667 0.96666667 0.96666667 0.96666667 1.        ] 
#  cross_val_score :  0.9733
# GradientBoostingClassifier [0.96666667 0.96666667 0.9        0.96666667 1.        ] 
#  cross_val_score :  0.96
# HistGradientBoostingClassifier [0.96666667 0.96666667 0.83333333 0.96666667 1.        ] 
#  cross_val_score :  0.9467
# KNeighborsClassifier [0.96666667 1.         0.93333333 0.96666667 1.        ]
#  cross_val_score :  0.9733
# LabelPropagation [0.96666667 0.96666667 0.93333333 0.96666667 1.        ] 
#  cross_val_score :  0.9667
# LabelSpreading [0.96666667 0.96666667 0.93333333 0.96666667 1.        ]
#  cross_val_score :  0.9667
# LinearDiscriminantAnalysis [1.         1.         0.96666667 0.93333333 1.        ] 
#  cross_val_score :  0.98
# LinearSVC [1.         1.         0.93333333 0.9        1.        ] 
#  cross_val_score :  0.9667
# LogisticRegression [0.96666667 1.         0.93333333 0.96666667 1.        ] 
#  cross_val_score :  0.9733
# LogisticRegressionCV [0.93333333 1.         0.93333333 0.93333333 1.        ] 
#  cross_val_score :  0.96
# MLPClassifier [1.         1.         0.96666667 0.96666667 1.        ] 
#  cross_val_score :  0.9867
# MultiOutputClassifier 은 안나온 놈!!
# MultinomialNB [1.         0.96666667 0.9        0.9        1.        ]
#  cross_val_score :  0.9533
# NearestCentroid [0.9        0.93333333 0.86666667 0.93333333 0.96666667]
#  cross_val_score :  0.92
# NuSVC [0.96666667 0.96666667 0.96666667 0.93333333 1.        ] 
#  cross_val_score :  0.9667
# OneVsOneClassifier 은 안나온 놈!!
# OneVsRestClassifier 은 안나온 놈!!
# OutputCodeClassifier 은 안나온 놈!!
# PassiveAggressiveClassifier [0.76666667 0.83333333 0.76666667 0.73333333 0.66666667] 
#  cross_val_score :  0.7533
# Perceptron [0.66666667 0.7        0.76666667 0.83333333 0.66666667]
#  cross_val_score :  0.7267
# QuadraticDiscriminantAnalysis [1.         1.         0.96666667 0.93333333 1.        ]
#  cross_val_score :  0.98
# RadiusNeighborsClassifier [0.93333333 1.         0.93333333 0.93333333 1.        ]
#  cross_val_score :  0.96
# RandomForestClassifier [0.96666667 0.96666667 0.9        0.96666667 1.        ] 
#  cross_val_score :  0.96
# RidgeClassifier [0.76666667 0.83333333 0.83333333 0.76666667 0.9       ]
#  cross_val_score :  0.82
# RidgeClassifierCV [0.76666667 0.83333333 0.83333333 0.76666667 0.9       ] 
#  cross_val_score :  0.82
# SGDClassifier [1.         0.73333333 0.93333333 0.43333333 0.76666667]
#  cross_val_score :  0.7733
# SVC [0.96666667 0.96666667 0.96666667 0.93333333 1.        ] 
#  cross_val_score :  0.9667
# StackingClassifier 은 안나온 놈!!
# VotingClassifier 은 안나온 놈!!