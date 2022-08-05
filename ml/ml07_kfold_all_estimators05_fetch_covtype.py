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
        
        y_predict = model.predict(x_test)
        acc = accuracy_score(y_test, y_predict)
        print(name, '의 정답률 : ', acc)
    except :
        # continue    
        print(name, '은 안나온 놈!!')  
      


# 모델의 개수 :  41
# AdaBoostClassifier 의 정답률 :  0.4998666127380532
# BaggingClassifier 의 정답률 :  0.9605603986127724
# BernoulliNB 의 정답률 :  0.6316876500606697
# CalibratedClassifierCV 의 정답률 :  0.6679001402717658
# CategoricalNB 은 안나온 놈!!
# ClassifierChain 은 안나온 놈!!
# ComplementNB 은 안나온 놈!!
# DecisionTreeClassifier 의 정답률 :  0.9397691970086831
# DummyClassifier 의 정답률 :  0.4873626326342693
# ExtraTreeClassifier 의 정답률 :  0.8599003468068811
# ExtraTreesClassifier 의 정답률 :  0.9523936559297093
# GaussianNB 의 정답률 :  0.459075927471752
# GaussianProcessClassifier 은 안나온 놈!!
# GradientBoostingClassifier 의 정답률 :  0.772510176157242
# HistGradientBoostingClassifier 의 정답률 :  0.7801519754223213
# KNeighborsClassifier 의 정답률 :  0.968331282324897
# LabelPropagation 은 안나온 놈!!
# LabelSpreading 은 안나온 놈!!
# LinearDiscriminantAnalysis 의 정답률 :  0.6794747123568239
# LinearSVC 의 정답률 :  0.514306859547516
# LogisticRegression 의 정답률 :  0.6203540356100962
# LogisticRegressionCV 의 정답률 :  0.6700859702417321
# MLPClassifier 의 정답률 :  0.7479927368484462
# MultiOutputClassifier 은 안나온 놈!!
# MultinomialNB 은 안나온 놈!!
# NearestCentroid 의 정답률 :  0.1944183885097631
# NuSVC 은 안나온 놈!!
# OneVsOneClassifier 은 안나온 놈!!
# OneVsRestClassifier 은 안나온 놈!!
# OutputCodeClassifier 은 안나온 놈!!
# PassiveAggressiveClassifier 의 정답률 :  0.5640043716599399
# Perceptron 의 정답률 :  0.5282221629389947
# QuadraticDiscriminantAnalysis 의 정답률 :  0.08322504582497871
# RadiusNeighborsClassifier 은 안나온 놈!!
# RandomForestClassifier 의 정답률 :  0.9546225140486907
# RidgeClassifier 의 정답률 :  0.7004294209271705
# RidgeClassifierCV 의 정답률 :  0.7004380265569736
# SGDClassifier 의 정답률 :  0.3678820684491795      