from sklearn import datasets
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np  

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
             train_size=0.7, random_state=20)


# print(x.shape, y.shape)

print(datasets.feature_names)
print(datasets.DESCR)

#2. 모델구성
model = Sequential()
model.add(Dense(40, input_dim=8,activation='selu'))
model.add(Dense(45))
model.add(Dense(50,activation='selu'))
model.add(Dense(55))
model.add(Dense(60,activation='selu'))
model.add(Dense(65))
model.add(Dense(70,activation='selu'))
model.add(Dense(75))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=30, validation_split=0.43)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

# R2 0.55 ~ 0.6 

# loss :  0.6084719300270081
# r2스코어 :  0.5540387047623216