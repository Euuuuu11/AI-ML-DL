#1. 데이터
import numpy as np
x = np.array([1,2,3])
y = np.array([1,2,3])
print(x.shape, y.shape)
#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(4, input_dim=1))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(32))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x, y, epochs=18)

#4. 평가, 예측
los = model.evaluate(x, y)
print('los :', los)

result = model.predict(x)
print('4의 예측값 :', result)

# loss : 3.907985046680551e-14
# 4의 예측값 : [[4.000001]]

