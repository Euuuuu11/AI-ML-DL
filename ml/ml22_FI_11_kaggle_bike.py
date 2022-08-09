# 실습 
# 피처임포턴스가 전체 중요도애서 하위 20~25% 컬럼들을 제거하여
# 데이터셋 재구성 후
# 각 모델별로 돌려서 결과 도출

# 기존 모델결과와 비교

#2. 모델구성
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

# 결과비교
# 1. DecisionTreeClassifier
# 기존 acc
# 컬럼 삭제 후 acc


import sklearn as datasets
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd

#1. 데이터
path = './_data/kaggle_bike/'
train_set = pd.read_csv(path + 'train.csv')
test_set = pd.read_csv(path + 'test.csv')
# print(train_set.shape)  # (10886, 12)
# print(test_set.shape)   # (6493, 9)
# train_set.info() # 데이터 온전한지 확인.
train_set['datetime'] = pd.to_datetime(train_set['datetime']) 

train_set['year'] = train_set['datetime'].dt.year  
train_set['month'] = train_set['datetime'].dt.month
train_set['day'] = train_set['datetime'].dt.day
train_set['hour'] = train_set['datetime'].dt.hour
train_set.drop(['datetime', 'day', 'year'], inplace=True, axis=1)
# train_set['month'] = train_set['month'].astype('category')
# train_set['hour'] = train_set['hour'].astype('category')
train_set = pd.get_dummies(train_set, columns=['season','weather'])
train_set.drop(['casual', 'registered'], inplace=True, axis=1)
train_set.drop('atemp', inplace=True, axis=1)

test_set['datetime'] = pd.to_datetime(test_set['datetime'])
test_set['month'] = test_set['datetime'].dt.month
test_set['hour'] = test_set['datetime'].dt.hour
# test_set['month'] = test_set['month'].astype('category')
# test_set['hour'] = test_set['hour'].astype('category')
test_set = pd.get_dummies(test_set, columns=['season','weather'])
drop_feature = ['datetime', 'atemp']
test_set.drop(drop_feature, inplace=True, axis=1)

x = train_set.drop(['count'], axis=1)
y = train_set['count']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
           train_size=0.8, shuffle=True, random_state=123)

#2. 모델구성
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

model1 = DecisionTreeRegressor()
model2 = RandomForestRegressor()
model3 = GradientBoostingRegressor()
model4 = XGBRegressor()

#3. 훈련
model1.fit(x_train, y_train)
model2.fit(x_train, y_train)
model3.fit(x_train, y_train)
model4.fit(x_train, y_train)


#4. 평가, 예측
result = model1.score(x_test, y_test)
print('model.score : ', result)

from sklearn.metrics import r2_score

y_predict = model1.predict(x_test)
acc = r2_score(y_test, y_predict)
print(model1,':',model1.feature_importances_)   # 전체 feature를 다 사용 안해도 된다.

y_predict = model2.predict(x_test)
acc2 = r2_score(y_test, y_predict)
print(model2,':',model2.feature_importances_) 

y_predict = model3.predict(x_test)
acc3 = r2_score(y_test, y_predict)
print(model3,':',model3.feature_importances_) 

y_predict = model4.predict(x_test)
acc4 = r2_score(y_test, y_predict)
print(model4,':',model4.feature_importances_)

print('DecisionTreeRegressor_score : ', acc)
print('RandomForestRegressor_score : ', acc2)
print('GradientBoostingRegressor_score : ', acc3)
print('XGBRegressor_score : ', acc4)

# DecisionTreeRegressor_score :  0.729721657892531
# RandomForestRegressor_score :  0.8517434183079354
# GradientBoostingRegressor_score :  0.7820786715570607
# XGBRegressor_score :  0.8673276732653832