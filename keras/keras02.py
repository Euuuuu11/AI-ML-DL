#1. 데이터
import numpy as np
x = np.array([1,2,3,5,4])
y = np.array([1,2,3,4,5])

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(4, input_dim=1))
model.add(Dense(8))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(64))
model.add(Dense(16))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mae',optimizer='adam')
model.fit(x, y, epochs=100)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss :', loss)

result = model.predict([6])
print('6의 예측값 :', result)

# loss : 0.41006460785865784
# 6의 예측값 : [[5.9183025]]