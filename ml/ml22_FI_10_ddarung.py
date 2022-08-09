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
path = './_data/ddarung/'
train_set = pd.read_csv(path + 'train.csv', 
                        index_col=0) 
test_set = pd.read_csv(path + 'test.csv', 
                       index_col=0)

train_set =  train_set.dropna()
test_set = test_set.fillna(test_set.mean())

x = train_set.drop(['count'], axis=1) 
y = train_set['count']


# x = x.drop(x.columns[[3]], axis=1)
# x = np.delete(x, 1, axis=1)
# print(x.shape)

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

# DecisionTreeRegressor_score :  0.6484104703809197
# RandomForestRegressor_score :  0.79924395959853
# GradientBoostingRegressor_score :  0.7652474546175402
# XGBRegressor_score :  0.7752495207146002