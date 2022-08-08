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
# result = model.score(x_test, y_test)
# print('model.score : ', result)

from sklearn.metrics import r2_score
y_predict = model1.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('accuracy_score : ', r2)

print(model1,':',model1.feature_importances_)   # 

import matplotlib.pyplot as plt

def plot_feature_importances(model):
    n_feature = datasets.data.shape[1]
    plt.barh(np.arange(n_feature), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_feature), datasets.feature_names)
    plt.xlabel('Feature Importances')
    plt.ylabel('Feature')
    plt.ylim(-1, n_feature)
    plt.title(model)

model5 ='XGBRegressor()'

plt.subplot(2, 2, 1)
plot_feature_importances(model1) 
plt.title(model1)

plt.subplot(2, 2, 2)
plot_feature_importances(model2) 
plt.title(model2)

plt.subplot(2, 2, 3)
plot_feature_importances(model3) 
plt.title(model3)

plt.subplot(2, 2, 4)
plot_feature_importances(model4) 
plt.title(model5)


plt.show()   

