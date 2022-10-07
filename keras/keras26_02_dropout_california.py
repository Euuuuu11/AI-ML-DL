from sklearn import datasets
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import numpy as np  
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
             train_size=0.8, random_state=72)


# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler = MaxAbsScaler()
# scaler = RobustScaler()

# scaler.fit(x_train)
# #print(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# print(x.shape, y.shape)

print(datasets.feature_names)
print(datasets.DESCR)

#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=8,activation='relu'))
model.add(Dense(80))
model.add(Dropout(0.3))
model.add(Dense(80,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(60,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(60,activation='relu'))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam',metrics=['mae'])

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
date= datetime.datetime.now()      # 2022-07-07 17:22:07.702644
date = date.strftime("%m%d_%H%M")  # 0707_1723
print(date)

filepath = './_ModelCheckpoint/k26_2/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

es = EarlyStopping(monitor='val_loss', patience=100, mode='min', 
              verbose=1, restore_best_weights=True) 
mcp = ModelCheckpoint(monitor='val_loss', mode='auto',verbose=1,
                      save_best_only=True,filepath= "".join([filepath,'k26_',date, '_', filename]))

model.fit(x_train, y_train, epochs=1000, validation_split=0.2,
                 batch_size=105, verbose=1, callbacks=[es,mcp])

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)



# dropout 적용 후 
# loss :  0.45644935965538025
# r2스코어 :  0.6515499823536177