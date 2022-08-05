from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.svm import LinearSVR # 레거시한 리니어 모델 사용
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score, r2_score
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold, cross_val_score

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target


x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.8, shuffle=True, random_state=66)

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
        model.fit(x_train, y_train)
        
        scores = cross_val_score(model, x_test, y_test, cv=5)  
        print(name , scores, '\n cross_val_score : ', round(np.mean(scores), 4))
    except :
        # continue    
        print(name, '은 안나온 놈!!')   



# 모델의 개수 :  54
# ARDRegression [0.82396169 0.3838684  0.88438543 0.66273631 0.64361981] 
#  cross_val_score :  0.6797
# AdaBoostRegressor [0.8412335  0.25586274 0.89095633 0.90757707 0.80534188] 
#  cross_val_score :  0.7402
# BaggingRegressor [0.92818358 0.0778364  0.85847808 0.92793053 0.88959246] 
#  cross_val_score :  0.7364
# BayesianRidge [0.7712672  0.58513076 0.85497802 0.69648908 0.7617276 ]
#  cross_val_score :  0.7339
# CCA [0.72190268 0.47987366 0.90296387 0.44452953 0.53135231] 
#  cross_val_score :  0.6161
# DecisionTreeRegressor [ 0.74344834 -0.02166231  0.89022089  0.74265584  0.53353136]
#  cross_val_score :  0.5776
# DummyRegressor [-0.00692484 -0.05438896 -0.00936267 -0.26152284 -0.07561897]
#  cross_val_score :  -0.0816
# ElasticNet [0.74238599 0.58344717 0.77957467 0.60960574 0.67008996] 
#  cross_val_score :  0.677
# ElasticNetCV [0.72420889 0.56557834 0.76244409 0.59855984 0.66518682] 
#  cross_val_score :  0.6632
# ExtraTreeRegressor [0.73670145 0.57528453 0.12332625 0.73581162 0.3834872 ]
#  cross_val_score :  0.5109
# ExtraTreesRegressor [0.888919   0.43349625 0.95074971 0.91196052 0.88771159] 
#  cross_val_score :  0.8146
# GammaRegressor [-0.0073639  -0.07187563 -0.01150679 -0.19655987 -0.10345897]
#  cross_val_score :  -0.0782
# GaussianProcessRegressor [-4.00392398 -8.79292879 -6.52356004 -5.83236905 -9.41075827] 
#  cross_val_score :  -6.9127
# GradientBoostingRegressor [0.87235628 0.2715951  0.85438693 0.92115292 0.89250634] 
#  cross_val_score :  0.7624
# HistGradientBoostingRegressor [0.82949305 0.40510731 0.74246375 0.84482184 0.8020194 ] 
#  cross_val_score :  0.7248
# HuberRegressor [0.5336631  0.60936244 0.72720243 0.72709329 0.52622536] 
#  cross_val_score :  0.6247
# IsotonicRegression 은 안나온 놈!!
# KNeighborsRegressor [ 0.1901187  -0.73772457  0.43611696  0.02711086  0.52618686]
#  cross_val_score :  0.0884
# KernelRidge [0.8426833  0.58392288 0.88281113 0.70579414 0.78541979] 
#  cross_val_score :  0.7601
# Lars [ 0.81688824 -0.25868302  0.88813006  0.71414023  0.7128921 ] 
#  cross_val_score :  0.5747
# LarsCV [0.86914245 0.51288305 0.74881751 0.71817148 0.7036309 ] 
#  cross_val_score :  0.7105
# Lasso [0.71325742 0.56523944 0.78294984 0.60494229 0.706734  ]
#  cross_val_score :  0.6746
# LassoCV [0.71411677 0.57202581 0.81138379 0.65530948 0.73255297] 
#  cross_val_score :  0.6971
# LassoLars [-0.00692484 -0.05438896 -0.00936267 -0.26152284 -0.07561897]
#  cross_val_score :  -0.0816
# LassoLarsCV [0.86173225 0.50880712 0.89005979 0.71817148 0.7036309 ] 
#  cross_val_score :  0.7365
# LassoLarsIC [0.87146514 0.50907668 0.86503785 0.65499923 0.64380781] 
#  cross_val_score :  0.7089
# LinearRegression [0.81957603 0.55718902 0.88813006 0.71414023 0.7128921 ]
#  cross_val_score :  0.7384
# LinearSVR [-0.17730167  0.6345571   0.20955711 -0.88675309 -0.03776209] 
#  cross_val_score :  -0.0515
# MLPRegressor [  0.16166456  -0.135078   -11.66734539   0.47049038  -2.12508925] 
#  cross_val_score :  -2.6591
# MultiOutputRegressor 은 안나온 놈!!
# MultiTaskElasticNet 은 안나온 놈!!
# MultiTaskElasticNetCV 은 안나온 놈!!
# MultiTaskLasso 은 안나온 놈!!
# MultiTaskLassoCV 은 안나온 놈!!
# NuSVR [ 0.2226558  -0.18055713  0.21227109  0.31421468  0.08569615]
#  cross_val_score :  0.1309
# OrthogonalMatchingPursuit [0.83729988 0.4393126  0.69317582 0.63458746 0.70894252]
#  cross_val_score :  0.6627
# OrthogonalMatchingPursuitCV [0.75621687 0.39626676 0.69317582 0.61934159 0.52325786] 
#  cross_val_score :  0.5977
# PLSCanonical [-1.39099016 -3.46113922 -0.56132294 -7.00954566 -3.02830406] 
#  cross_val_score :  -3.0903
# PLSRegression [0.88340496 0.52894135 0.91063426 0.71732326 0.38546465]
#  cross_val_score :  0.6852
# PassiveAggressiveRegressor [-0.04540033 -2.08840055 -1.33385615 -0.56913775  0.23315287]
#  cross_val_score :  -0.7607
# PoissonRegressor [0.84773469 0.61706485 0.81140585 0.83306482 0.80050857] 
#  cross_val_score :  0.782
# RANSACRegressor [0.68674188 0.36118972 0.74518674 0.77579068 0.60193719] 
#  cross_val_score :  0.6342
# RadiusNeighborsRegressor [nan nan nan nan nan]
#  cross_val_score :  nan
# RandomForestRegressor [0.8839901  0.20529573 0.91137848 0.9364705  0.86565791] 
#  cross_val_score :  0.7606
# RegressorChain 은 안나온 놈!!
# Ridge [0.8380158  0.58421738 0.88296711 0.70633754 0.76532162]
#  cross_val_score :  0.7554
# RidgeCV [0.8380158  0.57175381 0.88296711 0.70633754 0.76532162]
#  cross_val_score :  0.7529
# SGDRegressor [-5.35257829e+25 -1.32905546e+27 -6.61022850e+25 -2.85009206e+27
#  -2.17075979e+26]
#  cross_val_score :  -9.031703140247179e+26
# SVR [ 0.1926156  -0.24891728  0.15766504  0.27638784 -0.05710693] 
#  cross_val_score :  0.0641
# StackingRegressor 은 안나온 놈!!