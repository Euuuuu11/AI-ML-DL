#캐글 바이크

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score

#1. 데이터

path = './_data/kaggle_bike/'
train_set = pd.read_csv(path + 'train.csv')
test_set = pd.read_csv(path + 'test.csv')
# print(train_set.shape)  # (10886, 12)
# print(test_set.shape)   # (6493, 9)
# train_set.info() # 데이터 온전한지 확인.
train_set['datetime'] = pd.to_datetime(train_set['datetime']) 
#datetime은 날짜와 시간을 나타내는 정보이므로 DTYPE을 datetime으로 변경.
#세부 날짜별 정보를 보기 위해 날짜 데이터를 년도,월,일, 시간으로 나눈다.
train_set['year'] = train_set['datetime'].dt.year  # 분과 초는 모든값이 0이므로 추가x
train_set['month'] = train_set['datetime'].dt.month
train_set['day'] = train_set['datetime'].dt.day
train_set['hour'] = train_set['datetime'].dt.hour
train_set.drop(['datetime', 'day', 'year'], inplace=True, axis=1)
train_set['month'] = train_set['month'].astype('category')
train_set['hour'] = train_set['hour'].astype('category')
train_set = pd.get_dummies(train_set, columns=['season','weather'])
train_set.drop(['casual', 'registered'], inplace=True, axis=1)
train_set.drop('atemp', inplace=True, axis=1)

test_set['datetime'] = pd.to_datetime(test_set['datetime'])
test_set['month'] = test_set['datetime'].dt.month
test_set['hour'] = test_set['datetime'].dt.hour
test_set['month'] = test_set['month'].astype('category')
test_set['hour'] = test_set['hour'].astype('category')
test_set = pd.get_dummies(test_set, columns=['season','weather'])
drop_feature = ['datetime', 'atemp']
test_set.drop(drop_feature, inplace=True, axis=1)

x = train_set.drop(['count'], axis=1)
y = train_set['count']


# print(x.shape) # (10886, 15)
# print(y.shape) # (10886, )

x_train, x_test, y_train, y_test = train_test_split(x,y,
             train_size=0.8, shuffle=True, random_state=777)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)
print(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x.shape,y.shape) # ((10886, 15) (10886,)
print(x_train.shape,x_test.shape) # (8708, 15) (2178, 15)



#2. 모델구성
from sklearn.svm import LinearSVR, SVR
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression, LinearRegression # LinearRegression 회귀 
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

model = RandomForestRegressor()

#3.4. 컴파일, 훈련, 평가, 예측
# scores = cross_val_score(model, x, y, cv=kfold)
scores = cross_val_score(model, x_train, y_train, cv=5)         # 둘다 가능
 
print('R2 : ', scores, '\n cross_val_score : ', round(np.mean(scores), 4))

y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
print(y_predict)


print(y_test)
R2 = r2_score(y_test, y_predict)
print('cross_val_predict R2 : ', R2)


# cross_val_predict R2 :  0.8200144547188092