
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

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

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
        
        scores = cross_val_score(model, x_test, y_test, cv=5)  
        print(name , scores, '\n cross_val_score : ', round(np.mean(scores), 4))
    except :
        # continue    
        print(name, '은 안나온 놈!!')    

# 모델의 개수 :  54
# ARDRegression [0.68143814 0.48319739 0.50274371 0.51784088 0.54929701] 
#  cross_val_score :  0.5469
# AdaBoostRegressor [0.70134041 0.61406082 0.55805287 0.51555351 0.60109861] 
#  cross_val_score :  0.598
# BaggingRegressor [0.71398065 0.75765805 0.61055872 0.54927369 0.62206547] 
#  cross_val_score :  0.6507
# BayesianRidge [0.67332043 0.48254233 0.49309019 0.5329946  0.5660301 ] 
#  cross_val_score :  0.5496
# CCA [ 0.43861287 -0.01649493 -0.25857318 -0.13378135  0.36427569] 
#  cross_val_score :  0.0788
# DecisionTreeRegressor [0.68689197 0.54032247 0.3363227  0.35730298 0.50857347] 
#  cross_val_score :  0.4859
# DummyRegressor [-7.09043257e-04 -9.74002688e-02 -8.06075966e-05 -6.51870283e-03
#  -2.76998715e-02]
#  cross_val_score :  -0.0265
# ElasticNet [0.54534987 0.40133978 0.44872378 0.51704163 0.45675682]
#  cross_val_score :  0.4738
# ElasticNetCV [0.66176356 0.47457205 0.49012492 0.55620479 0.55529838] 
#  cross_val_score :  0.5476
# ExtraTreeRegressor [0.43771766 0.21560834 0.30140558 0.39795658 0.355591  ]
#  cross_val_score :  0.3417
# ExtraTreesRegressor [0.82414795 0.76268604 0.60193079 0.66774685 0.67407408] 
#  cross_val_score :  0.7061
# GammaRegressor [0.36029108 0.21855024 0.25498571 0.34470353 0.30622967] 
#  cross_val_score :  0.297
# GaussianProcessRegressor [ 0.0602406  -0.09531195 -0.13252221 -0.29491664  0.53940005] 
#  cross_val_score :  0.0154
# GradientBoostingRegressor [0.82173079 0.68777862 0.62495864 0.63465863 0.59896071] 
#  cross_val_score :  0.6736
# HistGradientBoostingRegressor [0.79174016 0.67202998 0.56878471 0.56011492 0.54829482] 
#  cross_val_score :  0.6282
# HuberRegressor [0.67919372 0.50421787 0.48500225 0.54476621 0.51396586] 
#  cross_val_score :  0.5454
# IsotonicRegression 은 안나온 놈!!
# KNeighborsRegressor [0.66501553 0.35666085 0.40560353 0.447694   0.55442747] 
#  cross_val_score :  0.4859
# KernelRidge [-0.89700765 -1.70625736 -1.82231759 -1.16519831 -1.29369632] 
#  cross_val_score :  -1.3769
# Lars [0.68443964 0.49009355 0.49687129 0.34244778 0.56712341] 
#  cross_val_score :  0.5162
# LarsCV [0.6796337  0.48730036 0.49687129 0.4372116  0.56226851] 
#  cross_val_score :  0.5327
# Lasso [0.66418939 0.47714073 0.49364441 0.54675109 0.54399414]
#  cross_val_score :  0.5451
# LassoCV [0.68048578 0.48627393 0.49705759 0.52423348 0.56739914] 
#  cross_val_score :  0.5511
# LassoLars [0.50780466 0.40400724 0.48878468 0.5123808  0.37960621]
#  cross_val_score :  0.4585
# LassoLarsCV [0.67952642 0.48730036 0.50075851 0.52163353 0.56226851] 
#  cross_val_score :  0.5503
# LassoLarsIC [0.68439169 0.49185868 0.49687129 0.53093699 0.54232099]
#  cross_val_score :  0.5493
# LinearRegression [0.68443964 0.49009355 0.49687129 0.51295914 0.56712341] 
#  cross_val_score :  0.5503
# LinearSVR [0.50700049 0.38795281 0.3029644  0.4440607  0.29704201]
#  cross_val_score :  0.3878
# MLPRegressor [-0.24097414 -0.22209785 -0.75729971 -0.27934563 -0.47117002] 
#  cross_val_score :  -0.3942
# MultiOutputRegressor 은 안나온 놈!!
# MultiTaskElasticNet 은 안나온 놈!!
# MultiTaskElasticNetCV 은 안나온 놈!!
# MultiTaskLasso 은 안나온 놈!!
# MultiTaskLassoCV 은 안나온 놈!!
# NuSVR [0.16247562 0.1697638  0.12142298 0.14217079 0.09200349] 
#  cross_val_score :  0.1376
# OrthogonalMatchingPursuit [0.4522365  0.36304497 0.40845095 0.32954704 0.31703756]
#  cross_val_score :  0.3741
# OrthogonalMatchingPursuitCV [0.67909045 0.47098921 0.50084547 0.48848676 0.50107124] 
#  cross_val_score :  0.5281
# PLSCanonical [-0.11846978 -0.66552674 -1.11770989 -0.64776601 -0.03061519]
#  cross_val_score :  -0.516
# PLSRegression [0.66761878 0.43862511 0.5271247  0.53908526 0.57498767] 
#  cross_val_score :  0.5495
# PassiveAggressiveRegressor [0.64513751 0.45394815 0.45037525 0.53623902 0.4947443 ]
#  cross_val_score :  0.5161
# PoissonRegressor [0.73578463 0.50324509 0.47695809 0.56402115 0.58643284] 
#  cross_val_score :  0.5733
# RANSACRegressor [0.49816315 0.42880408 0.44867677 0.52541644 0.42029799] 
#  cross_val_score :  0.4643
# RadiusNeighborsRegressor [nan nan nan nan nan]
#  cross_val_score :  nan
# RandomForestRegressor [0.79455063 0.74229691 0.61097465 0.61710692 0.64460417] 
#  cross_val_score :  0.6819
# RegressorChain 은 안나온 놈!!
# Ridge [0.68122317 0.48734343 0.49544827 0.52362253 0.56846846]
#  cross_val_score :  0.5512
# RidgeCV [0.68122317 0.48734343 0.49544827 0.5141991  0.56846846] 
#  cross_val_score :  0.5493
# SGDRegressor [0.67710224 0.48389743 0.48889611 0.52722925 0.56503854] 
#  cross_val_score :  0.5484
# SVR [0.22199984 0.1982144  0.19596186 0.22973958 0.18500824] 
#  cross_val_score :  0.2062
# StackingRegressor 은 안나온 놈!!
# TheilSenRegressor [0.64337765 0.4576116  0.44878402 0.53876364 0.53987128] 
#  cross_val_score :  0.5257
# TransformedTargetRegressor [0.68443964 0.49009355 0.49687129 0.51295914 0.56712341]
#  cross_val_score :  0.5503
# TweedieRegressor [0.46901947 0.34243881 0.39710105 0.45095791 0.3886339 ] 
#  cross_val_score :  0.4096
# VotingRegressor 은 안나온 놈!!