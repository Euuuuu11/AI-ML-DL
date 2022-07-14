# dropout 과적합 방지
# 평가, 예측엔 전체노드 적용

from tabnanny import verbose
from sklearn.datasets import load_boston
from tensorflow.python.keras.models import Sequential,  load_model
from tensorflow.python.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D, LSTM  
import numpy as np  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

#1. 데이터
datasets = load_boston()
x, y = datasets.data, datasets.target 
print(x.shape, y.shape)     # (506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(x,y,
             train_size=0.8, shuffle=True, random_state=32)


scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(404, 13, 1)
x_test = x_test.reshape(102, 13, 1)
print(x_train.shape)             


# 2. 모델구성
model = Sequential()
model.add(LSTM(170, return_sequences=True, activation= 'relu' ,input_shape = (13,1)))    
model.add(LSTM(90, return_sequences=False, activation = 'relu'))
model.add(Dense(60, activation = 'relu'))
model.add(Dense(20, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(1))
model.summary()

import time
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
date= datetime.datetime.now()      # 2022-07-07 17:22:07.702644
date = date.strftime("%m%d_%H%M")  # 0707_1723
print(date)

filepath = './_ModelCheckpoint/k26_1/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' # 04d 4자리수 까지 .4f 소수점 네자리까지

es = EarlyStopping(monitor='val_loss', patience=10, mode='min', 
              verbose=1, restore_best_weights=True) 
mcp = ModelCheckpoint(monitor='val_loss', mode='auto',verbose=1,
                      save_best_only=True,filepath= "".join([filepath,'k26_',date, '_', filename]))

start_time = time.time()
print(start_time)


hist = model.fit(x_train, y_train,
          epochs=30, batch_size=1,validation_split=0.2,
           verbose=1, callbacks=[es,mcp])

end_time = time.time() - start_time

# model.save('./_save/keras24_3_save_model.h5')
# model = load_model('./_ModelCheckpoint/keras24_ModelCheckpoint.hdf5')

# 4. 평가, 예측
print('================== 1. 기본 출력 =====================================')
loss = model.evaluate(x_test, y_test)
print("loss : ", loss) 


y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

# loss :  20.561464309692383
# r2스코어 :  0.7726058281758217

# loss :  20.444448471069336
# r2스코어 :  0.7738999183222043

# dropout 적용 후 
# loss :  20.24698829650879
# r2스코어 :  0.7760836586088403

# loss :  35.56769943237305
# r2스코어 :  0.6066482207423389