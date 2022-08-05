import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense
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
datasets = load_digits()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.8, shuffle=True, random_state=68)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

#2. 모델구성
allAlogrithms = all_estimators(type_filter='classifier')
# allAlogrithms = all_estimators(type_filter='regressor')

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

# 모델의 개수 :  41
# AdaBoostClassifier [0.20833333 0.23611111 0.23611111 0.30555556 0.19444444] 
#  cross_val_score :  0.2361
# BaggingClassifier [0.86111111 0.81944444 0.88888889 0.875      0.875     ] 
#  cross_val_score :  0.8639
# BernoulliNB [0.81944444 0.79166667 0.83333333 0.84722222 0.83333333] 
#  cross_val_score :  0.825
# CalibratedClassifierCV [0.97222222 0.94444444 0.93055556 0.88888889 0.94444444] 
#  cross_val_score :  0.9361
# CategoricalNB [  nan   nan   nan 0.875   nan] 
#  cross_val_score :  nan
# ClassifierChain 은 안나온 놈!!
# ComplementNB [0.75       0.76388889 0.77777778 0.77777778 0.81944444]
#  cross_val_score :  0.7778
# DecisionTreeClassifier [0.76388889 0.66666667 0.72222222 0.76388889 0.70833333] 
#  cross_val_score :  0.725
# DummyClassifier [0.125 0.125 0.125 0.125 0.125] 
#  cross_val_score :  0.125
# ExtraTreeClassifier [0.66666667 0.625      0.69444444 0.70833333 0.69444444]
#  cross_val_score :  0.6778
# ExtraTreesClassifier [0.97222222 0.94444444 0.94444444 0.95833333 0.98611111] 
#  cross_val_score :  0.9611
# GaussianNB [0.90277778 0.81944444 0.88888889 0.83333333 0.83333333] 
#  cross_val_score :  0.8556
# GaussianProcessClassifier [0.06944444 0.06944444 0.06944444 0.08333333 0.08333333] 
#  cross_val_score :  0.075
# GradientBoostingClassifier [0.91666667 0.77777778 0.88888889 0.93055556 0.90277778] 
#  cross_val_score :  0.8833
# HistGradientBoostingClassifier [0.94444444 0.875      0.95833333 0.90277778 0.93055556] 
#  cross_val_score :  0.9222
# KNeighborsClassifier [0.95833333 0.97222222 0.95833333 0.93055556 0.95833333] 
#  cross_val_score :  0.9556
# LabelPropagation [0.125      0.11111111 0.11111111 0.125      0.125     ] 
#  cross_val_score :  0.1194
# LabelSpreading [0.125      0.11111111 0.11111111 0.125      0.125     ] 
#  cross_val_score :  0.1194
# LinearDiscriminantAnalysis [0.93055556 0.91666667 0.875      0.93055556 0.93055556] 
#  cross_val_score :  0.9167
# LinearSVC [0.94444444 0.90277778 0.90277778 0.88888889 0.91666667] 
#  cross_val_score :  0.9111
# LogisticRegression [0.97222222 0.94444444 0.95833333 0.93055556 0.94444444] 
#  cross_val_score :  0.95
# LogisticRegressionCV [0.95833333 0.93055556 0.95833333 0.93055556 0.95833333] 
#  cross_val_score :  0.9472
# MLPClassifier [0.95833333 0.94444444 0.93055556 0.88888889 0.94444444] 
#  cross_val_score :  0.9333
# MultiOutputClassifier 은 안나온 놈!!
# MultinomialNB [0.93055556 0.90277778 0.90277778 0.875      0.90277778]
#  cross_val_score :  0.9028
# NearestCentroid [0.91666667 0.88888889 0.90277778 0.875      0.93055556] 
#  cross_val_score :  0.9028
# NuSVC [0.95833333 0.94444444 0.94444444 0.95833333 0.97222222] 
#  cross_val_score :  0.9556
# OneVsOneClassifier 은 안나온 놈!!
# OneVsRestClassifier 은 안나온 놈!!
# OutputCodeClassifier 은 안나온 놈!!
# PassiveAggressiveClassifier [0.97222222 0.93055556 0.93055556 0.93055556 0.93055556] 
#  cross_val_score :  0.9389
# Perceptron [0.93055556 0.90277778 0.86111111 0.90277778 0.91666667] 
#  cross_val_score :  0.9028
# QuadraticDiscriminantAnalysis [0.25       0.29166667 0.31944444 0.13888889 0.36111111] 
#  cross_val_score :  0.2722
# RadiusNeighborsClassifier [nan nan nan nan nan]
#  cross_val_score :  nan
# RandomForestClassifier [0.95833333 0.90277778 0.94444444 0.91666667 0.97222222] 
#  cross_val_score :  0.9389
# RidgeClassifier [0.90277778 0.88888889 0.84722222 0.875      0.88888889]
#  cross_val_score :  0.8806
# RidgeClassifierCV [0.90277778 0.88888889 0.84722222 0.875      0.90277778] 
#  cross_val_score :  0.8833
# SGDClassifier [0.95833333 0.86111111 0.91666667 0.91666667 0.93055556] 
#  cross_val_score :  0.9167
# SVC [0.98611111 0.97222222 0.95833333 0.97222222 0.97222222] 
#  cross_val_score :  0.9722
# StackingClassifier 은 안나온 놈!!
# VotingClassifier 은 안나온 놈!!