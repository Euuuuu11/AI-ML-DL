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
        
        y_predict = model.predict(x_test)
        acc = accuracy_score(y_test, y_predict)
        print(name, '의 정답률 : ', acc)
    except :
        # continue    
        print(name, '은 안나온 놈!!')  
        
# 모델의 개수 :  41
# AdaBoostClassifier 의 정답률 :  0.8333333333333334
# BaggingClassifier 의 정답률 :  0.8777777777777778
# BernoulliNB 의 정답률 :  0.7666666666666667
# CalibratedClassifierCV 의 정답률 :  0.6444444444444445
# CategoricalNB 의 정답률 :  0.8333333333333334
# ClassifierChain 은 안나온 놈!!
# ComplementNB 의 정답률 :  0.6888888888888889
# DecisionTreeClassifier 의 정답률 :  0.8444444444444444
# DummyClassifier 의 정답률 :  0.6
# ExtraTreeClassifier 의 정답률 :  0.7
# ExtraTreesClassifier 의 정답률 :  0.7555555555555555
# GaussianNB 의 정답률 :  0.8111111111111111
# GaussianProcessClassifier 의 정답률 :  0.6222222222222222
# GradientBoostingClassifier 의 정답률 :  0.8666666666666667
# HistGradientBoostingClassifier 의 정답률 :  0.8777777777777778
# KNeighborsClassifier 의 정답률 :  0.7111111111111111
# LabelPropagation 의 정답률 :  0.5888888888888889
# LabelSpreading 의 정답률 :  0.5888888888888889
# LinearDiscriminantAnalysis 의 정답률 :  0.8
# LinearSVC 의 정답률 :  0.6555555555555556
# LogisticRegression 의 정답률 :  0.7777777777777778
# LogisticRegressionCV 의 정답률 :  0.8222222222222222
# MLPClassifier 의 정답률 :  0.5444444444444444
# MultiOutputClassifier 은 안나온 놈!!
# MultinomialNB 의 정답률 :  0.7
# NearestCentroid 의 정답률 :  0.5666666666666667
# NuSVC 의 정답률 :  0.7444444444444445
# OneVsOneClassifier 은 안나온 놈!!
# OneVsRestClassifier 은 안나온 놈!!
# OutputCodeClassifier 은 안나온 놈!!
# PassiveAggressiveClassifier 의 정답률 :  0.6333333333333333
# Perceptron 의 정답률 :  0.4777777777777778
# QuadraticDiscriminantAnalysis 의 정답률 :  0.7666666666666667
# RadiusNeighborsClassifier 은 안나온 놈!!
# RandomForestClassifier 의 정답률 :  0.8222222222222222
# RidgeClassifier 의 정답률 :  0.8
# RidgeClassifierCV 의 정답률 :  0.8
# SGDClassifier 의 정답률 :  0.6555555555555556
# SVC 의 정답률 :  0.7
# StackingClassifier 은 안나온 놈!!
# VotingClassifier 은 안나온 놈!!