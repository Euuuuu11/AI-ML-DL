import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense
import pandas as pd
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
datasets = fetch_covtype()
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

# [예외처리] 에러가 떴을 때 무시하고, 넘어가겠다. 
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
# AdaBoostClassifier [0.63284712 0.62170302 0.62458586 0.58377797 0.52878657] 
#  cross_val_score :  0.5983
# BaggingClassifier [0.90516759 0.90766318 0.90899703 0.91054217 0.9098537 ] 
#  cross_val_score :  0.9084
# BernoulliNB [0.62320898 0.62523127 0.63512758 0.62951807 0.62564544] 
#  cross_val_score :  0.6277
# CalibratedClassifierCV [0.65324212 0.67165785 0.67118454 0.65636833 0.67048193] 
#  cross_val_score :  0.6646
# CategoricalNB 은 안나온 놈!!
# ClassifierChain 은 안나온 놈!!
# ComplementNB 은 안나온 놈!!
# DecisionTreeClassifier [0.86769072 0.86528118 0.86721742 0.8707401  0.86493115] 
#  cross_val_score :  0.8672
# DummyClassifier [0.48737146 0.48737146 0.48737146 0.4873494  0.4873494 ] 
#  cross_val_score :  0.4874
# ExtraTreeClassifier [0.79247881 0.78520718 0.80383804 0.78967298 0.79242685] 
#  cross_val_score :  0.7927
# ExtraTreesClassifier [0.91063207 0.91355794 0.91045996 0.91148881 0.91230637] 
#  cross_val_score :  0.9117
# GaussianNB [0.45626264 0.45806979 0.45686502 0.45391566 0.46105852] 
#  cross_val_score :  0.4572
# GaussianProcessClassifier 은 안나온 놈!!
# GradientBoostingClassifier [0.76877071 0.77402005 0.77105116 0.76923408 0.7697074 ] 
#  cross_val_score :  0.7706
# HistGradientBoostingClassifier [0.81166903 0.82294221 0.83744245 0.82375215 0.83390706] 
#  cross_val_score :  0.8259
# KNeighborsClassifier [0.91028785 0.9106751  0.90714685 0.9095525  0.90722892] 
#  cross_val_score :  0.909
# LabelPropagation 은 안나온 놈!!
# LabelSpreading 은 안나온 놈!!
# LinearDiscriminantAnalysis [0.67686416 0.68060755 0.67729444 0.67878657 0.67654905] 
#  cross_val_score :  0.678
# LinearSVC [0.44382772 0.23880212 0.20442322 0.51359725 0.50611015] 
#  cross_val_score :  0.3814
# LogisticRegression [0.61425928 0.61838991 0.62282174 0.62091222 0.61807229] 
#  cross_val_score :  0.6189
# LogisticRegressionCV [0.67114152 0.66985069 0.66997978 0.67026678 0.67241824] 
#  cross_val_score :  0.6707
# MLPClassifier [0.71799836 0.76253173 0.74248096 0.60684165 0.74234079] 
#  cross_val_score :  0.7144
# MultiOutputClassifier 은 안나온 놈!!
# MultinomialNB 은 안나온 놈!!
# NearestCentroid [0.19117078 0.19259068 0.18669593 0.19509466 0.19883821] 
#  cross_val_score :  0.1929
# NuSVC 은 안나온 놈!!
# OneVsOneClassifier 은 안나온 놈!!
# OneVsRestClassifier 은 안나온 놈!!
# OutputCodeClassifier 은 안나온 놈!!
# PassiveAggressiveClassifier [0.56559528 0.51284368 0.51327396 0.30094664 0.03442341] 
#  cross_val_score :  0.3854
# Perceptron [0.49817134 0.07835291 0.39542188 0.45438898 0.55120482] 
#  cross_val_score :  0.3955
# QuadraticDiscriminantAnalysis [0.08502216 0.08631298 0.0839895  0.08179862 0.08265921] 
#  cross_val_score :  0.084
# RadiusNeighborsClassifier [nan nan nan nan nan] 
#  cross_val_score :  nan
# RandomForestClassifier [0.90667355 0.90947033 0.90826557 0.90813253 0.90950947] 
#  cross_val_score :  0.9084
# RidgeClassifier [0.69618347 0.70190611 0.70212125 0.70051635 0.69909639] 
#  cross_val_score :  0.7
# RidgeClassifierCV [0.69618347 0.70190611 0.70207822 0.70051635 0.69909639] 
#  cross_val_score :  0.7
# SGDClassifier [0.53835893 0.63994665 0.60475023 0.50154905 0.5475043 ] 
#  cross_val_score :  0.5664
# SVC [0.70216428 0.70040015 0.70461684 0.70520654 0.70021515] 
#  cross_val_score :  0.7025
# StackingClassifier 은 안나온 놈!!
# VotingClassifier 은 안나온 놈!!
                                               