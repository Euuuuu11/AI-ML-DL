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
import numpy as np

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

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

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
        
        scores = cross_val_score(model, x_test, y_test, cv=5)  
        print(name , scores, '\n cross_val_score : ', round(np.mean(scores), 4))
    except :
        # continue    
        print(name, '은 안나온 놈!!')    

# 모델의 개수 :  54
# ARDRegression [0.25969405 0.36253298 0.33442779 0.34946079 0.2862006 ] 
#  cross_val_score :  0.3185
# AdaBoostRegressor [0.57886139 0.62071632 0.64597881 0.53120827 0.6071246 ] 
#  cross_val_score :  0.5968
# BaggingRegressor [0.80171095 0.794091   0.84304025 0.80224663 0.77356884] 
#  cross_val_score :  0.8029
# BayesianRidge [0.25917365 0.36348803 0.34037598 0.34997105 0.28623417] 
#  cross_val_score :  0.3198
# CCA [-0.11430134  0.17875149  0.05977034  0.04567538  0.05670076] 
#  cross_val_score :  0.0453
# DecisionTreeRegressor [0.68447916 0.62252662 0.73797328 0.70926785 0.64178546] 
#  cross_val_score :  0.6792
# DummyRegressor [-2.67372639e-05 -5.01684987e-06 -2.01906916e-04 -5.87533133e-03
#  -4.25487404e-03]
#  cross_val_score :  -0.0021
# ElasticNet [0.22339457 0.26053693 0.25642754 0.26533993 0.21227445]
#  cross_val_score :  0.2436
# ElasticNetCV [0.26570153 0.35703925 0.33909287 0.34887554 0.2822404 ] 
#  cross_val_score :  0.3186
# ExtraTreeRegressor [0.6368056  0.65225257 0.71550578 0.58566096 0.54313332] 
#  cross_val_score :  0.6267
# ExtraTreesRegressor [0.81041672 0.79934842 0.83338432 0.82022024 0.80507059] 
#  cross_val_score :  0.8137
# GammaRegressor [0.12182336 0.14951552 0.14268076 0.14021234 0.11345493] 
#  cross_val_score :  0.1335
# GaussianProcessRegressor [-13.87909721 -19.50889224  -7.06481842 -11.31260479  -4.40738802] 
#  cross_val_score :  -11.2346
# GradientBoostingRegressor [0.73828255 0.77834833 0.80326647 0.78481067 0.77150876] 
#  cross_val_score :  0.7752
# HistGradientBoostingRegressor [0.82033559 0.81610591 0.85511716 0.84490436 0.78774491] 
#  cross_val_score :  0.8248
# HuberRegressor [0.2183675  0.34312534 0.28126437 0.3509548  0.24065053] 
#  cross_val_score :  0.2869
# IsotonicRegression 은 안나온 놈!!
# KNeighborsRegressor [0.42288691 0.47426135 0.50297361 0.41811722 0.4203125 ] 
#  cross_val_score :  0.4477
# KernelRidge [0.25787703 0.36330117 0.33936966 0.34928969 0.28552048] 
#  cross_val_score :  0.3191
# Lars [0.25319037 0.3632736  0.33931531 0.3487395  0.28503758] 
#  cross_val_score :  0.3179
# LarsCV [0.26521601 0.35886401 0.32457176 0.34935097 0.28523306] 
#  cross_val_score :  0.3166
# Lasso [0.26281573 0.36142558 0.33922736 0.34949955 0.28449474]
#  cross_val_score :  0.3195
# LassoCV [0.26118699 0.36335127 0.33988306 0.3496692  0.28616046] 
#  cross_val_score :  0.3201
# LassoLars [0.1738796  0.18746853 0.1854601  0.19453799 0.16416198]
#  cross_val_score :  0.1811
# LassoLarsCV [0.25852517 0.35886401 0.33699565 0.34935097 0.28463674] 
#  cross_val_score :  0.3177
# LassoLarsIC [0.26458596 0.35966918 0.33751636 0.34933292 0.28341908] 
#  cross_val_score :  0.3189
# LinearRegression [0.25745966 0.3632736  0.33931531 0.3487395  0.28592852] 
#  cross_val_score :  0.3189
# LinearSVR [0.17513404 0.26030718 0.21172534 0.29450356 0.16590304] 
#  cross_val_score :  0.2215
# MLPRegressor [0.29067868 0.39453053 0.37072278 0.38133064 0.30544104] 
#  cross_val_score :  0.3485
# MultiOutputRegressor 은 안나온 놈!!
# MultiTaskElasticNet 은 안나온 놈!!
# MultiTaskElasticNetCV 은 안나온 놈!!
# MultiTaskLasso 은 안나온 놈!!
# MultiTaskLassoCV 은 안나온 놈!!
# NuSVR [0.14423175 0.16088267 0.15683058 0.18708114 0.1252591 ] 
#  cross_val_score :  0.1549
# OrthogonalMatchingPursuit [0.10464355 0.16075857 0.16353357 0.15310136 0.10161683] 
#  cross_val_score :  0.1367
# OrthogonalMatchingPursuitCV [0.25197136 0.35836477 0.32727113 0.34884554 0.27940813] 
#  cross_val_score :  0.3132
# PLSCanonical [-0.56560798 -0.3455377  -0.32005858 -0.46965818 -0.41734273] 
#  cross_val_score :  -0.4236
# PLSRegression [0.25369076 0.35555042 0.3398302  0.34070178 0.27822646] 
#  cross_val_score :  0.3136
# PassiveAggressiveRegressor [0.18816908 0.31848812 0.19796875 0.33483417 0.20681312] 
#  cross_val_score :  0.2493
# PoissonRegressor [0.27672868 0.39113538 0.37391514 0.38696837 0.29828537] 
#  cross_val_score :  0.3454
# RANSACRegressor [0.12171758 0.17139841 0.18076099 0.17028602 0.00150295] 
#  cross_val_score :  0.1291
# RadiusNeighborsRegressor [-5.33821212e+31 -5.73709874e+31 -5.59957095e+31 -5.73663503e+31
#  -3.39740613e+31]
#  cross_val_score :  -5.161784594631281e+31
# RandomForestRegressor [0.80886993 0.80217486 0.85280064 0.82282056 0.78047531] 
#  cross_val_score :  0.8134
# RegressorChain 은 안나온 놈!!
# Ridge [0.25764181 0.36339111 0.33946585 0.34893911 0.28602322]
#  cross_val_score :  0.3191
# RidgeCV [0.25917042 0.36354227 0.34031615 0.34991307 0.28624437] 
#  cross_val_score :  0.3198
# SGDRegressor [0.25810998 0.36441493 0.34215398 0.34876005 0.28664353] 
#  cross_val_score :  0.32
# SVR [0.1113822  0.14558729 0.13381635 0.1933806  0.09402085] 
#  cross_val_score :  0.1356
# StackingRegressor 은 안나온 놈!!
# TheilSenRegressor [0.24971098 0.36877816 0.336601   0.34942929 0.28345503] 
#  cross_val_score :  0.3176
# TransformedTargetRegressor [0.25745966 0.3632736  0.33931531 0.3487395  0.28592852] 
#  cross_val_score :  0.3189
# TweedieRegressor [0.17537965 0.19870912 0.19698339 0.20267274 0.16253648] 
#  cross_val_score :  0.1873
# VotingRegressor 은 안나온 놈!!