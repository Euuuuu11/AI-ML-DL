from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np  
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
             train_size=0.7, random_state=66)

# print(x.shape, y.shape)  

print(datasets.feature_names) 
print(datasets.DESCR)


#2. 모델구성
model = Sequential()
model.add(Dense(128, input_dim=13,activation='selu'))
model.add(Dense(64))
model.add(Dense(64,activation='selu'))
model.add(Dense(32))
model.add(Dense(16,activation='selu'))
model.add(Dense(8))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=10,
          validation_split=0.43)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

# loss :  17.505720138549805
# r2스코어 :  0.78811021977448

# validation 사용 후
# loss :  14.096229553222656
# r2스코어 :  0.8293788022088331