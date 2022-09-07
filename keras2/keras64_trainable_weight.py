from telnetlib import SE
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

#1. 데이터
x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 3, 4, 5])

#2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

# model.summary()

# print(model.weights)
# print('============================'*5)
# print(model.trainable_weights)
# print('============================'*5)

# print(len(model.weights))               # 6
# print(len(model.trainable_weights))     # 6

model.trainable = False

# print(len(model.weights))               # 6
# print(len(model.trainable_weights))     # 0

# print("="*66)
# print(model.trainable_weights)

# model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x, y, batch_size=1, epochs=100)

y_predict = model.predict(x)
print(y_predict[:3])