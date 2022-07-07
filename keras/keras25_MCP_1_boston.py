from tabnanny import verbose
from sklearn.datasets import load_boston
from tensorflow.python.keras.models import Sequential,  load_model
from tensorflow.python.keras.layers import Dense
import numpy as np  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

#1. 데이터
datasets = load_boston()
x, y = datasets.data, datasets. target 

x_train, x_test, y_train, y_test = train_test_split(x,y,
             train_size=0.8, shuffle=True, random_state=32)
# print(x)
# print(y)
# print(x.shape, y.shape)  # (506, 13) (506, )

# print(datasets.feature_names) 
# print(datasets.DESCR)

scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# 2. 모델구성
model = Sequential()
model.add(Dense(64, input_dim=13))
model.add(Dense(32,activation="relu"))
model.add(Dense(16,activation="relu"))
model.add(Dense(1))
model.summary()

                
            
# RuntimeError: You must compile your model before training/testing. 
# Use `model.compile(optimizer, loss)`.   # 컴파일 부분은 시작 해줘야 돌아감.
             
                   
import time
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
date= datetime.datetime.now()      # 2022-07-07 17:22:07.702644
date = date.strftime("%m%d_%H%M")  # 0707_1723
print(date)

filepath = './_ModelCheckpoint/k25_1/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' # 04d 4자리수 까지 .4f 소수점 네자리까지

es = EarlyStopping(monitor='val_loss', patience=10, mode='min', 
              verbose=1, restore_best_weights=True) 
mcp = ModelCheckpoint(monitor='val_loss', mode='auto',verbose=1,
                      save_best_only=True,filepath= "".join([filepath,'k25_',date, '_', filename]))

start_time = time.time()
print(start_time)

hist = model.fit(x_train, y_train,
          epochs=100, batch_size=1,validation_split=0.2,
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
'''
print('================== 2. load_model 출력 =====================================')
model2 = load_model('./_save/keras24_3_save_model.h5')
loss2 = model2.evaluate(x_test, y_test)
print('loss2 : ', loss2)

y_predict2 = model2.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict2)
print('r2스코어 : ', r2)

print('================== 3. ModelCheckpoint 출력 =====================================')
model3 = load_model('./_ModelCheckpoint/keras24_ModelCheckpoint3.hdf5')
loss3 = model3.evaluate(x_test, y_test)
print('loss3 : ', loss3)

y_predict3 = model3.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict3)
print('r2스코어 : ', r2)



# ================== 1. 기본 출력 =====================================
# 4/4 [==============================] - 0s 1ms/step - loss: 18.0411
# loss :  18.041139602661133
# r2스코어 :  0.8004786496646705
# ================== 2. load_model 출력 =====================================
# 4/4 [==============================] - 0s 1ms/step - loss: 18.0411
# loss2 :  18.041139602661133
# r2스코어 :  0.8004786496646705
# ================== 3. ModelCheckpoint 출력 =====================================
# 4/4 [==============================] - 0s 2ms/step - loss: 18.0411
# loss3 :  18.041139602661133
# r2스코어 :  0.8004786496646705
'''