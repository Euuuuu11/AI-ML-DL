import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from tqdm import tqdm_notebook
 # 레거시한 리니어 모델 사용


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
                                                    train_size=0.8, shuffle=True, random_state=68)

n_splits = 5
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=66)

#2. 모델구성
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron, LogisticRegression # LogisticRegression 분류 모델 사용
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier # 가지치기 형식으로 결과값 도출, 분류형식
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

#3.4. 컴파일, 훈련, 평가, 예측
# scores = cross_val_score(model, x, y, cv=kfold)
scores = cross_val_score(model, x_train, y_train, cv=5)         # 둘다 가능
 
print('ACC : ', scores, '\n cross_val_score : ', round(np.mean(scores), 4))

y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
print(y_predict)


print(y_test)
acc = accuracy_score(y_test, y_predict)
print('cross_val_predict ACC : ', acc)

# Kflod 사용
# cross_val_predict ACC :  0.7555555555555555

# Stratified_Kfold 사용
# cross_val_predict ACC :  0.7666666666666667