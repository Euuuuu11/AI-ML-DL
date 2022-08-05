#캐글 바이크

import numpy as np
import pandas as pd
from sklearn import metrics
from tensorflow.python.keras.models import Sequential,  load_model
from tensorflow.python.keras.layers import Activation, Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout,LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score, r2_score
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold, cross_val_score

#1. 데이터

path = './_data/kaggle_bike/'
train_set = pd.read_csv(path + 'train.csv')
test_set = pd.read_csv(path + 'test.csv')
# print(train_set.shape)  # (10886, 12)
# print(test_set.shape)   # (6493, 9)
# train_set.info() # 데이터 온전한지 확인.
train_set['datetime'] = pd.to_datetime(train_set['datetime']) 
#datetime은 날짜와 시간을 나타내는 정보이므로 DTYPE을 datetime으로 변경.
#세부 날짜별 정보를 보기 위해 날짜 데이터를 년도,월,일, 시간으로 나눈다.
train_set['year'] = train_set['datetime'].dt.year  # 분과 초는 모든값이 0이므로 추가x
train_set['month'] = train_set['datetime'].dt.month
train_set['day'] = train_set['datetime'].dt.day
train_set['hour'] = train_set['datetime'].dt.hour
train_set.drop(['datetime', 'day', 'year'], inplace=True, axis=1)
train_set['month'] = train_set['month'].astype('category')
train_set['hour'] = train_set['hour'].astype('category')
train_set = pd.get_dummies(train_set, columns=['season','weather'])
train_set.drop(['casual', 'registered'], inplace=True, axis=1)
train_set.drop('atemp', inplace=True, axis=1)

test_set['datetime'] = pd.to_datetime(test_set['datetime'])
test_set['month'] = test_set['datetime'].dt.month
test_set['hour'] = test_set['datetime'].dt.hour
test_set['month'] = test_set['month'].astype('category')
test_set['hour'] = test_set['hour'].astype('category')
test_set = pd.get_dummies(test_set, columns=['season','weather'])
drop_feature = ['datetime', 'atemp']
test_set.drop(drop_feature, inplace=True, axis=1)

x = train_set.drop(['count'], axis=1)
y = train_set['count']


# print(x.shape) # (10886, 15)
# print(y.shape) # (10886, )

x_train, x_test, y_train, y_test = train_test_split(x,y,
             train_size=0.8, shuffle=True, random_state=777)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)
print(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x.shape,y.shape) # ((10886, 15) (10886,)
print(x_train.shape,x_test.shape) # (8708, 15) (2178, 15)

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
# ARDRegression 의 정답률 :  0.31956799885696285
# AdaBoostRegressor 의 정답률 :  0.5760734000368517
# BaggingRegressor 의 정답률 :  0.8425207545282287
# BayesianRidge 의 정답률 :  0.3190338431564449
# CCA 의 정답률 :  0.14328974852898524
# DecisionTreeRegressor 의 정답률 :  0.7282905637710548
# DummyRegressor 의 정답률 :  -0.00010994723183221922
# ElasticNet 의 정답률 :  0.2476017289120308
# ElasticNetCV 의 정답률 :  0.32114136568403695
# ExtraTreeRegressor 의 정답률 :  0.7302247111750519
# ExtraTreesRegressor 의 정답률 :  0.8551973941403287
# GammaRegressor 의 정답률 :  0.1859868660963342
# GaussianProcessRegressor 의 정답률 :  -911.8333565619369
# GradientBoostingRegressor 의 정답률 :  0.7747437266767074
# HistGradientBoostingRegressor 의 정답률 :  0.8653064661501968
# HuberRegressor 의 정답률 :  0.2837761764458585
# IsotonicRegression 은 안나온 놈!!
# KNeighborsRegressor 의 정답률 :  0.5991803650691621
# KernelRidge 의 정답률 :  0.31874091522815184
# Lars 의 정답률 :  0.31956956641205314
# LarsCV 의 정답률 :  0.3209902404995073
# Lasso 의 정답률 :  0.322491296888504
# LassoCV 의 정답률 :  0.3191955517415147
# LassoLars 의 정답률 :  -0.00010994723183221922
# LassoLarsCV 의 정답률 :  0.3191767410769094
# LassoLarsIC 의 정답률 :  0.31892598825068585
# LinearRegression 의 정답률 :  0.3187546767347773
# LinearSVR 의 정답률 :  0.2580887467741554
# MLPRegressor 의 정답률 :  0.5259864604101937
# MultiOutputRegressor 은 안나온 놈!!
# MultiTaskElasticNet 은 안나온 놈!!
# MultiTaskElasticNetCV 은 안나온 놈!!
# MultiTaskLasso 은 안나온 놈!!
# MultiTaskLassoCV 은 안나온 놈!!
# NuSVR 의 정답률 :  0.3106937118609061
# OrthogonalMatchingPursuit 의 정답률 :  0.1492573902275134
# OrthogonalMatchingPursuitCV 의 정답률 :  0.3188680077929762
# PLSCanonical 의 정답률 :  -0.3390427534095217
# PLSRegression 의 정답률 :  0.3147120684736393
# PassiveAggressiveRegressor 의 정답률 :  0.24519066089284902
# PoissonRegressor 의 정답률 :  0.3100343279410974
# RANSACRegressor 의 정답률 :  0.126869586411683
# RadiusNeighborsRegressor 의 정답률 :  -3.5043665858255334e+30
# RandomForestRegressor 의 정답률 :  0.8567805216834365
# RegressorChain 은 안나온 놈!!
# Ridge 의 정답률 :  0.31879447976582176
# RidgeCV 의 정답률 :  0.31909451662990795
# SGDRegressor 의 정답률 :  0.31839290603456083
# SVR 의 정답률 :  0.2905248797927362
# StackingRegressor 은 안나온 놈!!
# TheilSenRegressor 의 정답률 :  0.3167278296343399
# TransformedTargetRegressor 의 정답률 :  0.3187546767347773
# TweedieRegressor 의 정답률 :  0.19083065122694054
# VotingRegressor 은 안나온 놈!!