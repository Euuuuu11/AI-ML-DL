#캐글 바이크

import numpy as np
import pandas as pd
from sklearn import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

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

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)
print(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델구성
model = Sequential()
model.add(Dense(128, input_dim=15,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64))
model.add(Dropout(0.3))
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(8))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam', metrics=['mse'])

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
date= datetime.datetime.now()      # 2022-07-07 17:22:07.702644
date = date.strftime("%m%d_%H%M")  # 0707_1723
print(date)

filepath = './_ModelCheckpoint/k26_10/'
filename = '{epoch:04d}-{loss:.4f}.hdf5'

from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=100, mode='min', 
              verbose=1, restore_best_weights=True) 
mcp = ModelCheckpoint(monitor='loss', mode='auto',verbose=1,
                      save_best_only=True,filepath= "".join([filepath,'k26_',date, '_', filename]))

model.fit(x_train, y_train, epochs=100, batch_size=100, verbose=1, callbacks=[es,mcp])

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

# dropout 적용 후 
# loss :  70.77356719970703
# r2스코어 :  0.5761524272062224