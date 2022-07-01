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
             train_size=0.8, random_state=62)


# print(x.shape, y.shape)

print(datasets.feature_names)
print(datasets.DESCR)

#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=8,activation='selu'))
model.add(Dense(80))
model.add(Dense(80,activation='selu'))
model.add(Dense(60))
model.add(Dense(60,activation='selu'))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=100, mode='min', 
              verbose=1, restore_best_weights=True) 

hist = model.fit(x_train, y_train, epochs=105, validation_split=0.2,
                 batch_size=105, verbose=1, callbacks=[es])

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)


# R2 0.55 ~ 0.6 

# loss :  0.6084719300270081
# r2스코어 :  0.5540387047623216

# validation 사용 후
# loss :  0.5098468065261841
# r2스코어 :  0.6256723529290654

# EarlyStopping 후
# loss :  0.4342330992221832
# r2스코어 :  0.6648673057808461