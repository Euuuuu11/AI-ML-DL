from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np  
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets. target

x_train, x_test, y_train, y_test = train_test_split(x,y,
             train_size=0.7, shuffle=True, random_state=100)
# print(x)
# print(y)
# print(x.shape, y.shape)  # (506, 13) (506, )

print(datasets.feature_names) 
print(datasets.DESCR)


#2. 모델구성
model = Sequential()
model.add(Dense(11, input_dim=13))
model.add(Dense(22))
model.add(Dense(33))
model.add(Dense(44))
model.add(Dense(55))
model.add(Dense(66))
model.add(Dense(77))
model.add(Dense(88))
model.add(Dense(99))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

# loss :  17.505720138549805
# r2스코어 :  0.78811021977448