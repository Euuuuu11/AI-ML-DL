from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense
from sklearn.svm import LinearSVR # 레거시한 리니어 모델 사용
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score, r2_score
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold, cross_val_score

datasets = fetch_california_housing()
x = datasets.data
y = datasets.target


x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.7, shuffle=True, random_state=66)

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
# ARDRegression 의 정답률 :  0.5950451417085713
# AdaBoostRegressor 의 정답률 :  0.42120568616090426
# BaggingRegressor 의 정답률 :  0.7882938431732502
# BayesianRidge 의 정답률 :  0.6004227260166415
# CCA 의 정답률 :  0.565984358025825
# DecisionTreeRegressor 의 정답률 :  0.5953502278433038
# DummyRegressor 의 정답률 :  -0.0007832271331533747
# ElasticNet 의 정답률 :  0.4309559637448974
# ElasticNetCV 의 정답률 :  0.5945198387923485
# ExtraTreeRegressor 의 정답률 :  0.5773687615165632
# ExtraTreesRegressor 의 정답률 :  0.8159133407067418
# GammaRegressor 의 정답률 :  -0.0007832271331533747
# GaussianProcessRegressor 의 정답률 :  -2.837863291635716
# GradientBoostingRegressor 의 정답률 :  0.7926230798531358
# HistGradientBoostingRegressor 의 정답률 :  0.8368747531074239
# HuberRegressor 의 정답률 :  -6.835018664677069
# IsotonicRegression 은 안나온 놈!!
# KNeighborsRegressor 의 정답률 :  0.13188588139050017
# KernelRidge 의 정답률 :  0.535813020928104
# Lars 의 정답률 :  0.6001949284390906
# LarsCV 의 정답률 :  0.6039503639556894
# Lasso 의 정답률 :  0.2816450299323535
# LassoCV 의 정답률 :  0.598495640188932
# LassoLars 의 정답률 :  -0.0007832271331533747
# LassoLarsCV 의 정답률 :  0.6039503639556894
# LassoLarsIC 의 정답률 :  0.6012092775566569
# LinearRegression 의 정답률 :  0.6001949284390906
# LinearSVR 의 정답률 :  -5.177019771115873
# MLPRegressor 의 정답률 :  -0.33380288887532217
# MultiOutputRegressor 은 안나온 놈!!
# MultiTaskElasticNet 은 안나온 놈!!
# MultiTaskElasticNetCV 은 안나온 놈!!
# MultiTaskLasso 은 안나온 놈!!
# MultiTaskLassoCV 은 안나온 놈!!
# NuSVR 의 정답률 :  -0.002599078650999953
# OrthogonalMatchingPursuit 의 정답률 :  0.4861662015494945
# OrthogonalMatchingPursuitCV 의 정답률 :  0.6061292835338536
# PLSCanonical 의 정답률 :  0.35686745387634955
# PLSRegression 의 정답률 :  0.523820478691435
# PassiveAggressiveRegressor 의 정답률 :  -5.020270336318801
# PoissonRegressor 의 정답률 :  -0.0007832271331533747
# RANSACRegressor 의 정답률 :  -35.953131021645085
# RadiusNeighborsRegressor 은 안나온 놈!!
# RandomForestRegressor 의 정답률 :  0.8091282506958237
# RegressorChain 은 안나온 놈!!
# Ridge 의 정답률 :  0.6002646839899062
# RidgeCV 의 정답률 :  0.6002019515659932
# SGDRegressor 의 정답률 :  -5.563457686907089e+30
# SVR 의 정답률 :  -0.0370628287403556
# StackingRegressor 은 안나온 놈!!
# TheilSenRegressor 의 정답률 :  -30.419181442534807
# TransformedTargetRegressor 의 정답률 :  0.6001949284390906
# TweedieRegressor 의 정답률 :  0.49447485764038346
# VotingRegressor 은 안나온 놈!!           