import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm_notebook
 # 레거시한 리니어 모델 사용
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold, cross_val_score

#1. 데이터

path = './_data/kaggle_titanic/' # 경로 = .현재폴더 /하단
train_set = pd.read_csv(path + 'train.csv', # train.csv 의 데이터가 train set에 들어가게 됨
                        index_col=0) # 0번째 컬럼은 인덱스로 지정하는 명령
test_set = pd.read_csv(path + 'test.csv',
                       index_col=0)

# print(train_set)
# print(train_set.shape) # (891, 11) 원래 열이 12개지만, id를 인덱스로 제외하여 11개

# print(train_set.columns)
# print(train_set.info()) # 각 컬럼에 대한 디테일한 내용 출력 / null값(중간에 빠진 값) '결측치'
# print(train_set.describe())

print(test_set)
print(test_set.shape) # (418, 10) # 예측 과정에서 쓰일 예정


# 결측치 처리
print(train_set.isnull().sum()) # 각 컬럼당 null의 갯수 확인가능 -- age 177, cabin 687, embarked 2
# Survived      0
# Pclass        0
# Name          0
# Sex           0
# Age         177
# SibSp         0
# Parch         0
# Ticket        0
# Fare          0
# Cabin       687
# Embarked      2
# dtype: int64
train_set = train_set.fillna(train_set.median())
print(test_set.isnull().sum())
# Pclass        0
# Name          0
# Sex           0
# Age          86
# SibSp         0
# Parch         0
# Ticket        0
# Fare          1
# Cabin       327
# Embarked      0
# dtype: int64

drop_cols = ['Cabin']
train_set.drop(drop_cols, axis = 1, inplace =True)
test_set = test_set.fillna(test_set.mean())
train_set['Embarked'].fillna('S')
train_set = train_set.fillna(train_set.mean())

print(train_set) 
print(train_set.isnull().sum())

test_set.drop(drop_cols, axis = 1, inplace =True)
cols = ['Name','Sex','Ticket','Embarked']
for col in tqdm_notebook(cols):
    le = LabelEncoder()
    train_set[col]=le.fit_transform(train_set[col])
    test_set[col]=le.fit_transform(test_set[col])
    
x = train_set.drop(['Survived'],axis=1) #axis는 컬럼 
print(x) #(891, 9)
y = train_set['Survived']
print(y.shape) #(891,)

gender_submission = pd.read_csv(path + 'gender_submission.csv', #예측에서 쓰일 예정
                       index_col=0)

# print(pd.Series.value_counts()) 

x_train, x_test, y_train, y_test = train_test_split(x,y, 
                                                    train_size=0.9, shuffle=True, random_state=68)

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
# AdaBoostClassifier [0.61111111 0.66666667 0.61111111 0.83333333 0.72222222] 
#  cross_val_score :  0.6889
# BaggingClassifier [0.77777778 0.83333333 0.72222222 0.94444444 0.72222222] 
#  cross_val_score :  0.8
# BernoulliNB [0.72222222 0.61111111 0.88888889 0.83333333 0.66666667] 
#  cross_val_score :  0.7444
# CalibratedClassifierCV [0.66666667 0.61111111 0.61111111 0.61111111 0.55555556] 
#  cross_val_score :  0.6111
# CategoricalNB [       nan        nan        nan 0.94444444 0.66666667] 
#  cross_val_score :  nan
# ClassifierChain 은 안나온 놈!!
# ComplementNB [0.55555556 0.72222222 0.38888889 0.72222222 0.66666667] 
#  cross_val_score :  0.6111
# DecisionTreeClassifier [0.77777778 0.88888889 0.66666667 0.83333333 0.72222222] 
#  cross_val_score :  0.7778
# DummyClassifier [0.61111111 0.61111111 0.61111111 0.61111111 0.55555556]
#  cross_val_score :  0.6
# ExtraTreeClassifier [0.61111111 0.88888889 0.72222222 0.66666667 0.72222222] 
#  cross_val_score :  0.7222
# ExtraTreesClassifier [0.77777778 0.77777778 0.83333333 0.77777778 0.77777778] 
#  cross_val_score :  0.7889
# GaussianNB [0.72222222 0.77777778 0.66666667 0.83333333 0.77777778] 
#  cross_val_score :  0.7556
# GaussianProcessClassifier [0.55555556 0.66666667 0.55555556 0.66666667 0.61111111] 
#  cross_val_score :  0.6111
# GradientBoostingClassifier [0.72222222 0.94444444 0.66666667 0.83333333 0.72222222] 
#  cross_val_score :  0.7778
# HistGradientBoostingClassifier [0.66666667 0.83333333 0.83333333 0.88888889 0.83333333] 
#  cross_val_score :  0.8111
# KNeighborsClassifier [0.5        0.66666667 0.44444444 0.66666667 0.72222222] 
#  cross_val_score :  0.6
# LabelPropagation [0.61111111 0.61111111 0.66666667 0.66666667 0.55555556] 
#  cross_val_score :  0.6222
# LabelSpreading [0.61111111 0.61111111 0.66666667 0.66666667 0.55555556] 
#  cross_val_score :  0.6222
# LinearDiscriminantAnalysis [0.72222222 0.66666667 0.77777778 0.77777778 0.83333333] 
#  cross_val_score :  0.7556
# LinearSVC [0.38888889 0.61111111 0.38888889 0.44444444 0.44444444] 
#  cross_val_score :  0.4556
# LogisticRegression [0.72222222 0.72222222 0.61111111 0.83333333 0.66666667] 
#  cross_val_score :  0.7111
# LogisticRegressionCV [0.72222222 0.72222222 0.72222222 0.77777778 0.88888889] 
#  cross_val_score :  0.7667
# MLPClassifier [0.66666667 0.5        0.72222222 0.5        0.61111111] 
#  cross_val_score :  0.6
# MultiOutputClassifier 은 안나온 놈!!
# MultinomialNB [0.55555556 0.72222222 0.38888889 0.72222222 0.66666667]
#  cross_val_score :  0.6111
# NearestCentroid [0.38888889 0.55555556 0.5        0.61111111 0.55555556] 
#  cross_val_score :  0.5222
# NuSVC [0.5        0.66666667 0.5        0.88888889 0.66666667] 
#  cross_val_score :  0.6444
# OneVsOneClassifier 은 안나온 놈!!
# OneVsRestClassifier 은 안나온 놈!!
# OutputCodeClassifier 은 안나온 놈!!
# PassiveAggressiveClassifier [0.66666667 0.55555556 0.72222222 0.72222222 0.55555556]
#  cross_val_score :  0.6444
# Perceptron [0.33333333 0.72222222 0.72222222 0.44444444 0.44444444] 
#  cross_val_score :  0.5333
# QuadraticDiscriminantAnalysis [0.77777778 0.83333333 0.61111111 0.83333333 0.72222222] 
#  cross_val_score :  0.7556
# RadiusNeighborsClassifier [nan nan nan nan nan] 
#  cross_val_score :  nan
# RandomForestClassifier [0.72222222 0.88888889 0.66666667 0.83333333 0.72222222] 
#  cross_val_score :  0.7667
# RidgeClassifier [0.72222222 0.66666667 0.77777778 0.77777778 0.83333333]
#  cross_val_score :  0.7556
# RidgeClassifierCV [0.72222222 0.66666667 0.77777778 0.77777778 0.83333333] 
#  cross_val_score :  0.7556
# SGDClassifier [0.66666667 0.83333333 0.61111111 0.61111111 0.61111111] 
#  cross_val_score :  0.6667
# SVC [0.55555556 0.66666667 0.55555556 0.66666667 0.77777778] 
#  cross_val_score :  0.6444
# StackingClassifier 은 안나온 놈!!
# VotingClassifier 은 안나온 놈!!