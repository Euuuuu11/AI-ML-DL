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
# ARDRegression 의 정답률 :  0.8012569266997994
# AdaBoostRegressor 의 정답률 :  0.8960380193719849
# BaggingRegressor 의 정답률 :  0.9150434658873201
# BayesianRidge 의 정답률 :  0.7937918622384768
# CCA 의 정답률 :  0.7913477184424631
# DecisionTreeRegressor 의 정답률 :  0.8159958439593297
# DummyRegressor 의 정답률 :  -0.0005370164400797517
# ElasticNet 의 정답률 :  0.7338335519267194
# ElasticNetCV 의 정답률 :  0.7167760356856183
# ExtraTreeRegressor 의 정답률 :  0.8011743556393625
# ExtraTreesRegressor 의 정답률 :  0.9372178070602093
# GammaRegressor 의 정답률 :  -0.0005370164400797517
# GaussianProcessRegressor 의 정답률 :  -6.073105259620457
# GradientBoostingRegressor 의 정답률 :  0.9456255148709037
# HistGradientBoostingRegressor 의 정답률 :  0.9323597806119726
# HuberRegressor 의 정답률 :  0.751179780083514
# IsotonicRegression 은 안나온 놈!!
# KNeighborsRegressor 의 정답률 :  0.5900872726222293
# KernelRidge 의 정답률 :  0.833332549399853
# Lars 의 정답률 :  0.7746736096721593
# LarsCV 의 정답률 :  0.7981576314184005
# Lasso 의 정답률 :  0.7240751024070102
# LassoCV 의 정답률 :  0.7517507753137198
# LassoLars 의 정답률 :  -0.0005370164400797517
# LassoLarsCV 의 정답률 :  0.8127604328474289
# LassoLarsIC 의 정답률 :  0.8131423868817643
# LinearRegression 의 정답률 :  0.8111288663608656
# LinearSVR 의 정답률 :  0.6449514946396616
# MLPRegressor 의 정답률 :  0.5546047892605901
# MultiOutputRegressor 은 안나온 놈!!
# MultiTaskElasticNet 은 안나온 놈!!
# MultiTaskElasticNetCV 은 안나온 놈!!
# MultiTaskLasso 은 안나온 놈!!
# MultiTaskLassoCV 은 안나온 놈!!
# NuSVR 의 정답률 :  0.2594558622083819
# OrthogonalMatchingPursuit 의 정답률 :  0.5827617571381449
# OrthogonalMatchingPursuitCV 의 정답률 :  0.7861744773872901
# PLSCanonical 의 정답률 :  -2.2317079741425747
# PLSRegression 의 정답률 :  0.8027313142007888
# PassiveAggressiveRegressor 의 정답률 :  0.08639005815725143
# PoissonRegressor 의 정답률 :  0.8575614023948606
# RANSACRegressor 의 정답률 :  -0.791402559818764
# RadiusNeighborsRegressor 은 안나온 놈!!
# RandomForestRegressor 의 정답률 :  0.9165106178738819
# RegressorChain 은 안나온 놈!!
# Ridge 의 정답률 :  0.8098487632912241
# RidgeCV 의 정답률 :  0.8112529184583794
# SGDRegressor 의 정답률 :  -3.0214967918962e+26
# SVR 의 정답률 :  0.23474677555722312
# StackingRegressor 은 안나온 놈!!
# TheilSenRegressor 의 정답률 :  0.7717743846429955
# TransformedTargetRegressor 의 정답률 :  0.8111288663608656
# TweedieRegressor 의 정답률 :  0.7436268475903307
# VotingRegressor 은 안나온 놈!!