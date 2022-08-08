import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
       train_size=0.8, shuffle=True, random_state=1234)

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

#2. 모델
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import make_pipeline

# model = RandomForestRegressor()
model = make_pipeline(MinMaxScaler(), RandomForestRegressor())

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)
print('model.score : ', result)

# no pipeline
# model.score :  0.8157469028714444

# piperine
# model.score :  0.8092765869958356



