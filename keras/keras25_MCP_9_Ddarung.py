#데이콘 따릉이 문제풀이
import numpy as np
import pandas as pd
from sklearn import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

#1. 데이터
path = './_data/ddarung/'
train_set = pd.read_csv(path + 'train.csv', 
                        index_col=0) 


test_set = pd.read_csv(path + 'test.csv', 
                       index_col=0)



train_set =  train_set.dropna()

test_set = test_set.fillna(test_set.mean())


x = train_set.drop(['count'], axis=1) 
#

y = train_set['count']



x_train, x_test, y_train, y_test = train_test_split(x,y,
             train_size=0.75, shuffle=True, random_state=85)

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
model.add(Dense(100, input_dim=9,activation='relu'))
model.add(Dense(100))
model.add(Dense(100,activation='relu'))
model.add(Dense(100))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam', metrics=['mse'])

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
date= datetime.datetime.now()      # 2022-07-07 17:22:07.702644
date = date.strftime("%m%d_%H%M")  # 0707_1723
print(date)

filepath = './_ModelCheckpoint/k25_9/'
filename = '{epoch:04d}-{loss:.4f}.hdf5'

from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=100, mode='min', 
              verbose=1, restore_best_weights=True) 
mcp = ModelCheckpoint(monitor='loss', mode='auto',verbose=1,
                      save_best_only=True,filepath= "".join([filepath,'k25_',date, '_', filename]))

model.fit(x_train, y_train, epochs=100, batch_size=10, verbose=1, callbacks=[es,mcp])


#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)


#1. 스케일러 하기 전
#  loss :  31.814311981201172
#  r2스코어 : 0.6615527797335422

#2. MinMaxScaler 
# loss :  27.314002990722656
# r2스코어 :  0.72836367006165

#3. StandardScaler  
# loss :  28.994173049926758
# r2스코어 :   0.7079155699820482

#4. MaxAbsScaler 
# loss :  27.7861328125
# r2스코어 :  0.7310630314838201

#5. RobustScaler 
#  loss :  28.494075775146484
#  r2스코어 :   0.7268625643887887