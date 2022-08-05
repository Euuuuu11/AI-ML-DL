
#데이콘 따릉이 문제풀이
import numpy as np
import pandas as pd
from sklearn import metrics
from tensorflow.python.keras.models import Sequential,  load_model
from tensorflow.python.keras.layers import Activation, Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout,LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score, r2_score
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold, cross_val_score

#1. 데이터
path = './_data/ddarung/'
train_set = pd.read_csv(path + 'train.csv', 
                        index_col=0) 


test_set = pd.read_csv(path + 'test.csv', 
                       index_col=0)



train_set =  train_set.dropna()

test_set = test_set.fillna(test_set.mean())


x = train_set.drop(['count'], axis=1) 
#

y = train_set['count']



x_train, x_test, y_train, y_test = train_test_split(x,y,
             train_size=0.75, shuffle=True, random_state=85)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

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
# ARDRegression 의 정답률 :  0.5889267737623805
# AdaBoostRegressor 의 정답률 :  0.5156808812206619
# BaggingRegressor 의 정답률 :  0.7197832719178439
# BayesianRidge 의 정답률 :  0.5869176360056512
# CCA 의 정답률 :  0.1944709590245125
# DecisionTreeRegressor 의 정답률 :  0.4713306284344092
# DummyRegressor 의 정답률 :  -0.000635693745441257
# ElasticNet 의 정답률 :  0.5122013454994756
# ElasticNetCV 의 정답률 :  0.5813813149363498
# ExtraTreeRegressor 의 정답률 :  0.38422013783911435
# ExtraTreesRegressor 의 정답률 :  0.7528900252200791
# GammaRegressor 의 정답률 :  0.4500252525645435
# GaussianProcessRegressor 의 정답률 :  -0.23471697762809152
# GradientBoostingRegressor 의 정답률 :  0.7251789490736102
# HistGradientBoostingRegressor 의 정답률 :  0.7318527506070103
# HuberRegressor 의 정답률 :  0.5744534170342174
# IsotonicRegression 은 안나온 놈!!
# KNeighborsRegressor 의 정답률 :  0.638962207620777
# KernelRidge 의 정답률 :  -1.2127512942170555
# Lars 의 정답률 :  0.5879603377840328
# LarsCV 의 정답률 :  0.5879603377840328
# Lasso 의 정답률 :  0.5770016795322664
# LassoCV 의 정답률 :  0.5877825220767784
# LassoLars 의 정답률 :  0.33942007470874
# LassoLarsCV 의 정답률 :  0.5879603377840328
# LassoLarsIC 의 정답률 :  0.588389268205311
# LinearRegression 의 정답률 :  0.5879603377840328
# LinearSVR 의 정답률 :  0.5097666150164776
# MLPRegressor 의 정답률 :  0.5576532959673897
# MultiOutputRegressor 은 안나온 놈!!
# MultiTaskElasticNet 은 안나온 놈!!
# MultiTaskElasticNetCV 은 안나온 놈!!
# MultiTaskLasso 은 안나온 놈!!
# MultiTaskLassoCV 은 안나온 놈!!
# NuSVR 의 정답률 :  0.42328015695730214
# OrthogonalMatchingPursuit 의 정답률 :  0.32833895504970434
# OrthogonalMatchingPursuitCV 의 정답률 :  0.5746458859063959
# PLSCanonical 의 정답률 :  -0.527361969782626
# PLSRegression 의 정답률 :  0.5830802819339325
# PassiveAggressiveRegressor 의 정답률 :  0.5427187172404988
# PoissonRegressor 의 정답률 :  0.6239389159045684
# RANSACRegressor 의 정답률 :  0.5191613314137944
# RadiusNeighborsRegressor 은 안나온 놈!!
# RandomForestRegressor 의 정답률 :  0.7461238404523163
# RegressorChain 은 안나온 놈!!
# Ridge 의 정답률 :  0.5876603284689087
# RidgeCV 의 정답률 :  0.5876603284689351
# SGDRegressor 의 정답률 :  0.5835739773264919
# SVR 의 정답률 :  0.4153046602834609
# StackingRegressor 은 안나온 놈!!
# TheilSenRegressor 의 정답률 :  0.5557536444807369
# TransformedTargetRegressor 의 정답률 :  0.5879603377840328
# TweedieRegressor 의 정답률 :  0.44789013104662756
# VotingRegressor 은 안나온 놈!!