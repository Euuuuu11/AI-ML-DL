# CNN
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D # 이미지
# https://nagadi.tistory.com/21
model  = Sequential()
# model.add(Dense(units=10, input_shape=(3, ))) #(batch_size, input_dim) 
# (input_dim + bias) * units = summary Param 갯수 (Dense모델)

model.add(Conv2D(filters=10, kernel_size=(2, 2), # 이미지 자르는 구격 # 'filters = output node 갯수'
                 input_shape=(5, 5, 1))) # N 5x5 1(흑백) 3(컬러) # (batch_size, rows, cols, channels))
model.add(Conv2D(7, (2,2), activation="relu")) # 이미지는 0 ~ 255
model.summary()
# (kernel_size * channels + bias) + filters  = summary Param 갯수 (CNN모델)



