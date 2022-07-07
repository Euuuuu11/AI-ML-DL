from sklearn import datasets
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input
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
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)
#print(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# print(x.shape, y.shape)

print(datasets.feature_names)
print(datasets.DESCR)

#2. 모델구성
input1 = Input(shape=(8,))
dense1 = Dense(80)(input1)
dense2 = Dense(80,activation='relu')(dense1)
dense3 = Dense(60,activation='relu')(dense2)
dense4 = Dense(60,activation='relu')(dense3)
dense5 = Dense(10)(dense4)
output1 = Dense(1)(dense5)

model = Model(inputs=input1, outputs=output1)

model.load_weights("./_save/keras23_8_save_weights_california.h5")
# model.save("./_save/keras23_8_save_model_california.h5")

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam',metrics=['mae'])

# from tensorflow.python.keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor='val_loss', patience=100, mode='min', 
#               verbose=1, restore_best_weights=True) 

# model.fit(x_train, y_train, epochs=100, validation_split=0.2,
#                  batch_size=105, verbose=1, callbacks=[es])

# model = load_model("./_save/keras23_8_save_model_california.h5")
# model.save_weights("./_save/keras23_8_save_weights_california.h5")

 
#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)



# loss :  0.43560951948165894
# r2스코어 :  0.6674590012250166

# loss :  [4.173460960388184, 1.7367783784866333]
# r2스코어 :  -2.1859883252688235