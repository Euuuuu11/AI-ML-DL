from subprocess import call
from tabnanny import verbose
from sklearn.datasets import load_boston
from tensorflow.python.keras.models import Sequential, Model,load_model
from tensorflow.python.keras.layers import Dense, Input
import numpy as np  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets. target

x_train, x_test, y_train, y_test = train_test_split(x,y,
             train_size=0.8, shuffle=True, random_state=66)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)
#print(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) #x_train이작업된 범위에 맞춰서 진행

# #2. 모델구성
input1 = Input(shape=(13,))
dense1 = Dense(10)(input1)
dense2 = Dense(80)(dense1)
dense3 = Dense(80, activation='relu')(dense2)
dense4 = Dense(60)(dense3)
dense5 = Dense(10, activation='relu')(dense4)
output1 = Dense(1)(dense5)
model = Model(inputs=input1, outputs=output1)

model.load_weights("./_save/keras23_7_save_weights_boston.h5")
# # model.save("./_save/keras23_7_save_model_boston.h5")

# import time
# #3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam',metrics=['mae'])

# from tensorflow.python.keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor='val_loss', patience=100, mode='min', 
#               verbose=1, restore_best_weights=True) 

# start_time = time.time()
# print(start_time)

# model.fit(x_train, y_train,
#           epochs=100, batch_size=1,validation_split=0.2,
#           verbose=1, callbacks=[es] )



# end_time = time.time() - start_time
# print("걸린시간 : ", end_time)

# model.save("./_save/keras23_7_save_model_boston.h5")
# model = load_model("./_save/keras23_7_save_model_boston.h5")
# model.save_weights("./_save/keras23_7_save_weights2_boston.h5")

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss) 

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

# loss :  10.007562637329102
# r2스코어 :  0.8802678881921752