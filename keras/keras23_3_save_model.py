from tabnanny import verbose
from sklearn.datasets import load_boston
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets. target

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
# print(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델구성
model = Sequential()
model.add(Dense(64, input_dim=13))
model.add(Dense(32,activation="relu"))
model.add(Dense(16,activation="relu"))
model.add(Dense(8,activation="relu"))
model.add(Dense(1))
model.summary()


# model.save("./_save/keras23_1_save_model.h5")

import time
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', 
              verbose=1, restore_best_weights=True) 


start_time = time.time()
print(start_time)

hist = model.fit(x_train, y_train,
          epochs=100, batch_size=1,validation_split=0.2,
           verbose=1, callbacks=[es])

end_time = time.time() - start_time

model.save("./_save/keras23_3_save_model.h5")

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss) 


y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

# loss :  21.24750518798828
# r2스코어 :  0.7650186946453363