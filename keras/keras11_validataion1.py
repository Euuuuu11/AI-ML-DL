from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np

#1. 데이터
x_train = np.array(range(1, 11))  # 훈련
y_train = np.array(range(1, 11))
x_test = np.array([11,12,13]) # 평가
y_test = np.array([11,12,13])
x_val = np.array([14,15,16])  # 검증
y_val = np.array([14,15,16])

#2. 모델구성
model = Sequential()
model.add(Dense(6, input_dim=1))
model.add(Dense(12))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train, epochs=100, batch_size=1,
          validation_data=(x_val, y_val))

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

result = model.predict([17])
print("17의 예측값 : ", result)






