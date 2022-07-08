# CNN
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten   # Conn2D = 이미지 
# https://nagadi.tistory.com/21
model  = Sequential()
# model.add(Dense(units=10, input_shape=(3, ))) #(batch_size, input_dim) 
# (input_dim + bias) * units = summary Param 갯수 (Dense모델)

model.add(Conv2D(filters=10, kernel_size=(3, 3), # 이미지 자르는 구격 # 'filters = output node 갯수' # 출력 (N, 6, 6, 10) 
                 input_shape=(8, 8, 1))) # N장 5x5 1(흑백) 3(컬러) # (batch_size, rows, cols, channels))
model.add(Conv2D(7, (2,2), activation="relu")) # 이미지는 0 ~ 255 # 출력 (N, 5, 5, 7)
# 4차원 -> 2차원 순서와 위치가 바뀌지 않게 쫙 핀다. 
model.add(Flatten())  # (N, 175)
model.add(Dense(32, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(10, activation="softmax"))

model.summary()
# (kernel_size * channels + bias) + filters  = summary Param 갯수 (CNN모델)



