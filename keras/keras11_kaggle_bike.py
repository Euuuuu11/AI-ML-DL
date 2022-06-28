#캐글 바이크

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

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


print(x.shape) # (10886, 15)
print(y.shape) # (10886, )

x_train, x_test, y_train, y_test = train_test_split(x,y,
             train_size=0.8, shuffle=True, random_state=66)

scaler = RobustScaler()  # 아주 동 떨어진 데이터를 제거
                         # 중간값과 사분위 값을 조정 # 전체 데이터와 아주 동떨어진 데이터 포인트에 영향을 받지않음.
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_set = scaler.transform(test_set)
#2. 모델구성
model = Sequential()
model.add(Dense(80, input_dim=15,activation='relu'))
model.add(Dropout(0.2))  # 입력 단위를 무작위로 0으로 
                         #설정하여 과적합을 방지하는 데 도움이 됩니다
model.add(Dense(50,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(20,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10,activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=5000, batch_size=100)

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict) : # def 함수를 만드는거 
    return np. sqrt(mean_squared_error(y_test, y_predict)) # 루트 씌움
rmse = RMSE(y_test, y_predict) 
print("RMSE : ", rmse)


############### 제출용 ####################
y_summit = model.predict(test_set)          
# print(y_summit.shape) # (6493, 1)

submission = pd.read_csv('C:\study\_data\kaggle_bike\samplesubmission.csv',index_col=0)
submission['count'] = y_summit
submission.to_csv('C:\study\_data\kaggle_bike\samplesubmission.csv', index=True)
