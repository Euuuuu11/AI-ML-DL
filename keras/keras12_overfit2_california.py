from sklearn import datasets
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Dropout
import numpy as np  
from sklearn.preprocessing import OneHotEncoder

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
             train_size=0.8, shuffle=True, random_state=66)


# print(x)
# print(y)
# print(x.shape, y.shape)  # (20640, 8) (20640, )

print(datasets.feature_names)
print(datasets.DESCR)

#2. 모델구성
model = Sequential()
model.add(Dense(40, input_dim=8,activation='selu'))
model.add(Dense(45))
model.add(Dropout(0.2))
model.add(Dense(50,activation='selu'))
model.add(Dropout(0.2))
model.add(Dense(55))
model.add(Dropout(0.2))
model.add(Dense(60,activation='selu'))
model.add(Dense(65))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train, y_train, epochs=500, validation_split=0.2, 
          batch_size=100, verbose=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

# print("========================")
# print(hist) 
# # <tensorflow.python.keras.callbacks.History object at 0x00000167B1367D60>
# print("========================")
# print(hist.history) # 키 밸류 형태로 'loss'와 'val_loss를 반환해준다.
# print("========================")
# print(hist.history['loss'])
# print("========================")
# print(hist.history['val_loss'])

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] =False

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], marker='.',  c = 'red', label = 'loss')
plt.plot(hist.history['val_loss'], marker='.',c = 'blue', label = 'val_loss')
plt.grid()
plt.title('이결바보')
plt.ylabel('loss')
plt.xlabel('epochs')
# plt.legend(loc='upper right')
plt.legend
plt.show()