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
from sklearn.model_selection import KFold, cross_val_score

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

# x_train, x_test, y_train, y_test = train_test_split(x, y,
#         train_size=0.8, shuffle=True, random_state=68)
#셔플을 잘 해주어야 데이터 분류에 오류가 없음
# print(y_train)
# print(y_test)
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

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
        
        scores = cross_val_score(model, x, y, cv=5)  
        print(name , scores, '\n cross_val_score : ', round(np.mean(scores), 4))
    except :
        # continue    
        print(name, '은 안나온 놈!!')    

# 모델의 개수 :  54
# ARDRegression [0.         0.85331976 0.         0.74868592 0.        ] 
#  cross_val_score :  0.3204
# AdaBoostRegressor [1.         0.99514292 0.         0.6999321  0.        ] 
#  cross_val_score :  0.539
# BaggingRegressor [1.    0.997 0.    0.853 0.   ] 
#  cross_val_score :  0.57
# BayesianRidge [0.         0.85157108 0.         0.75740574 0.        ]
#  cross_val_score :  0.3218
# CCA [0.         0.19120642 0.         0.57803088 0.        ] 
#  cross_val_score :  0.1538
# DecisionTreeRegressor [1.   0.85 0.   0.7  0.  ]
#  cross_val_score :  0.51
# DummyRegressor [ 0.    -3.125  1.    -3.125  0.   ]
#  cross_val_score :  -1.05
# ElasticNet [ 0.         -0.03117661  0.         -0.33139241  0.        ] 
#  cross_val_score :  -0.0725
# ElasticNetCV [0.         0.84215577 0.         0.75476055 0.        ] 
#  cross_val_score :  0.3194
# ExtraTreeRegressor [1.  1.  0.  0.7 0. ]
#  cross_val_score :  0.54
# ExtraTreesRegressor [0.       0.99619  0.       0.791125 0.      ] 
#  cross_val_score :  0.3575
# GammaRegressor [nan nan nan nan nan]
#  cross_val_score :  nan
# GaussianProcessRegressor [ 0.          0.46808604  0.         -7.84626383  0.        ]
#  cross_val_score :  -1.4756
# GradientBoostingRegressor [0.         0.95200461 0.         0.66398523 0.        ] 
#  cross_val_score :  0.3232
# HistGradientBoostingRegressor [0.         0.97157455 0.         0.76700205 0.        ] 
#  cross_val_score :  0.3477
# HuberRegressor [0.         0.8541555  0.         0.74813297 0.        ] 
#  cross_val_score :  0.3205
# IsotonicRegression [nan nan nan nan nan]
#  cross_val_score :  nan
# KNeighborsRegressor [1.    0.994 0.    0.778 0.   ] 
#  cross_val_score :  0.5544
# KernelRidge [0.         0.85063069 0.         0.75914501 0.        ]
#  cross_val_score :  0.322
# Lars [0.         0.85124923 0.         0.76155439 0.        ] 
#  cross_val_score :  0.3226
# LarsCV [0.         0.84968557 0.         0.76155439 0.        ] 
#  cross_val_score :  0.3222
# Lasso [ 0.         -1.5938856   0.         -1.59875162  0.        ] 
#  cross_val_score :  -0.6385
# LassoCV [0.         0.83919671 0.         0.75599553 0.        ] 
#  cross_val_score :  0.319
# LassoLars [ 0.    -3.125  1.    -3.125  0.   ]
#  cross_val_score :  -1.05
# LassoLarsCV [0.         0.84968557 0.         0.76155439 0.        ] 
#  cross_val_score :  0.3222
# LassoLarsIC [0.         0.83610871 0.         0.68952767 0.        ]
#  cross_val_score :  0.3051
# LinearRegression [0.         0.85124923 0.         0.76155439 0.        ]
#  cross_val_score :  0.3226
# LinearSVR [0.         0.85937395 0.         0.75202505 0.        ] 
#  cross_val_score :  0.3223
# MLPRegressor [0.         0.84876276 0.         0.69879719 0.        ] 
#  cross_val_score :  0.3095
# MultiOutputRegressor 은 안나온 놈!!
# MultiTaskElasticNet [nan nan nan nan nan]
#  cross_val_score :  nan
# MultiTaskElasticNetCV [nan nan nan nan nan]
#  cross_val_score :  nan
# MultiTaskLasso [nan nan nan nan nan] 
#  cross_val_score :  nan
# MultiTaskLassoCV [nan nan nan nan nan]
#  cross_val_score :  nan
# NuSVR [0.         0.88945501 0.         0.81939524 0.        ] 
#  cross_val_score :  0.3418
# OrthogonalMatchingPursuit [0.         0.85314976 0.         0.68055749 0.        ]
#  cross_val_score :  0.3067
# OrthogonalMatchingPursuitCV [0.         0.85314976 0.         0.73125824 0.        ] 
#  cross_val_score :  0.3169
# PLSCanonical [ 0.         -2.25620922  0.         -0.7813378   0.        ] 
#  cross_val_score :  -0.6075
# PLSRegression [0.         0.81275088 0.         0.65591203 0.        ]
#  cross_val_score :  0.2937
# PassiveAggressiveRegressor [0.         0.80937132 0.         0.29588422 0.        ]
#  cross_val_score :  0.2211
# PoissonRegressor [        nan  0.32375269        -inf -0.02127691        -inf] 
#  cross_val_score :  nan
# RANSACRegressor [0.         0.85656942 0.         0.75044417 0.        ] 
#  cross_val_score :  0.3214
# RadiusNeighborsRegressor [1.         0.82927354 0.         0.70402947 0.        ] 
#  cross_val_score :  0.5067
# RandomForestRegressor [1.      0.98125 0.      0.77245 0.     ] 
#  cross_val_score :  0.5507
# RegressorChain 은 안나온 놈!!
# Ridge [0.         0.85144794 0.         0.75273062 0.        ]
#  cross_val_score :  0.3208
# RidgeCV [0.         0.85144794 0.         0.75273062 0.        ] 
#  cross_val_score :  0.3208
# SGDRegressor [0.         0.79373085 0.         0.72588011 0.        ]
#  cross_val_score :  0.3039
# SVR [0.         0.88056099 0.         0.82555552 0.        ] 
#  cross_val_score :  0.3412
# StackingRegressor 은 안나온 놈!!
# TheilSenRegressor [0.         0.83440094 0.         0.57751421 0.        ] 
#  cross_val_score :  0.2824
# TransformedTargetRegressor [0.         0.85124923 0.         0.76155439 0.        ]
#  cross_val_score :  0.3226
# TweedieRegressor [      -inf 0.68196631       -inf 0.3610001        -inf]
#  cross_val_score :  -inf
# VotingRegressor 은 안나온 놈!!
