from unittest import result
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense
from sklearn.svm import LinearSVR # 레거시한 리니어 모델 사용
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score, r2_score
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
tf.random.set_seed(66)
# 웨이트의 난수

#1. 데이터
datasets = load_diabetes()
print(datasets.DESCR)
print(datasets.feature_names)
x = datasets['data']
y = datasets.target
print(x)
print(y)
print(x.shape, y.shape) # (150, 4) (150,)

# 원핫인코딩은 모델구성 전 데이터 전처리에서 진행
print("y의 라벨값 : ", np.unique(y)) # y의 라벨값 :  [0 1 2] (총 3개가 있다는 것을 알 수 있음)

# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)
# print(y)
# print(y.shape) #(150, 3)

x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.8, shuffle=True, random_state=68)
#셔플을 잘 해주어야 데이터 분류에 오류가 없음
# print(y_train)
# print(y_test)

#2. 모델구성
# allAlogrithms = all_estimators(type_filter='classifier')
allAlogrithms = all_estimators(type_filter='regressor')

# [('AdaBoostClassifier', <class 'sklearn.ensemble._weight_boosting.AdaBoostClassifier'>

print('allAlogrithms : ', allAlogrithms)    # 딕셔너리들이 list 형태로 묶여져있다.
print('모델의 개수 : ', len(allAlogrithms))  # 모델의 개수 :  41

# [예외처리] 에러가 떳을 때 무시하고, 넘어가겠다. 
for (name, algorithm) in allAlogrithms:
    try :
        model = algorithm()
        model.fit(x_train, y_train)
        
        y_predict = model.predict(x_test)
        r2 = r2_score(y_test, y_predict)
        print(name, '의 정답률 : ', r2)
    except :
        # continue    
        print(name, '은 안나온 놈!!')    

# 모델의 개수 :  54
# ARDRegression 의 정답률 :  0.9308518748569723
# AdaBoostRegressor 의 정답률 :  0.9529977002516952
# BaggingRegressor 의 정답률 :  0.978743961352657
# BayesianRidge 의 정답률 :  0.9291670947825628
# CCA 의 정답률 :  0.902046862113883
# DecisionTreeRegressor 의 정답률 :  0.9516908212560387
# DummyRegressor 의 정답률 :  -0.022644927536231707
# ElasticNet 의 정답률 :  0.6706043943895418
# ElasticNetCV 의 정답률 :  0.9281859761132686
# ExtraTreeRegressor 의 정답률 :  0.9516908212560387
# ExtraTreesRegressor 의 정답률 :  0.9733719806763285
# GammaRegressor 은 안나온 놈!!
# GaussianProcessRegressor 의 정답률 :  0.7015342105862833
# GradientBoostingRegressor 의 정답률 :  0.9667740784063744
# HistGradientBoostingRegressor 의 정답률 :  0.9647089309071
# HuberRegressor 의 정답률 :  0.9295132201158808
# IsotonicRegression 은 안나온 놈!!
# KNeighborsRegressor 의 정답률 :  0.9497584541062802
# KernelRidge 의 정답률 :  0.9277119059523082
# Lars 의 정답률 :  0.9311142723847698
# LarsCV 의 정답률 :  0.9311142723847698
# Lasso 의 정답률 :  0.37686053470114245
# LassoCV 의 정답률 :  0.9296792463135679
# LassoLars 의 정답률 :  -0.022644927536231707
# LassoLarsCV 의 정답률 :  0.9312521690774441
# LassoLarsIC 의 정답률 :  0.9119948872657565
# LinearRegression 의 정답률 :  0.9311142723847701
# LinearSVR 의 정답률 :  0.9374575395598228
# MLPRegressor 의 정답률 :  0.9213540064379683
# MultiOutputRegressor 은 안나온 놈!!
# MultiTaskElasticNet 은 안나온 놈!!
# MultiTaskElasticNetCV 은 안나온 놈!!
# MultiTaskLasso 은 안나온 놈!!
# MultiTaskLassoCV 은 안나온 놈!!
# NuSVR 의 정답률 :  0.9403477580098953
# OrthogonalMatchingPursuit 의 정답률 :  0.9216061763272163
# OrthogonalMatchingPursuitCV 의 정답률 :  0.9327775704123742
# PLSCanonical 의 정답률 :  0.5666429533491824
# PLSRegression 의 정답률 :  0.9317488844344691
# PassiveAggressiveRegressor 의 정답률 :  0.8386081575608235
# PoissonRegressor 의 정답률 :  0.776512821853325
# RANSACRegressor 의 정답률 :  0.9299674092528556
# RadiusNeighborsRegressor 의 정답률 :  0.9256993609647514
# RandomForestRegressor 의 정답률 :  0.9817246376811594
# RegressorChain 은 안나온 놈!!
# Ridge 의 정답률 :  0.9276613987944005
# RidgeCV 의 정답률 :  0.9276613987943834
# SGDRegressor 의 정답률 :  0.9225126075606517
# SVR 의 정답률 :  0.9424955665861705
# StackingRegressor 은 안나온 놈!!
# TheilSenRegressor 의 정답률 :  0.9098904204064733
# TransformedTargetRegressor 의 정답률 :  0.9311142723847701
# TweedieRegressor 의 정답률 :  0.8425179565465477
# VotingRegressor 은 안나온 놈!!
