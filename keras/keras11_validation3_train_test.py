from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array(range(1, 17))
y = np.array(range(1, 17))
# [실습] train_test_split로만 나눠라
x_train, x_test_val, y_train, y_test_val = train_test_split(
    x,y,train_size=0.625,random_state=66)

x_test, x_val, y_test, y_bal =  train_test_split(
    x_test_val,y_test_val,train_size=0.5,random_state=66)
# print(x_train,y_train,x_test,y_test,x_val,y_bal) 

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train, epochs=100, batch_size=1,
          validation_data=(x_val, y_bal))

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

result = model.predict([17])
print("17의 예측값 : ", result)







