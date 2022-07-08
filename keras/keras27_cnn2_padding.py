# CNN_padding
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D   # Conn2D = 이미지 



model  = Sequential()
# model.add(Dense(units=10, input_shape=(3, ))) #(batch_size, input_dim) 
# (input_dim + bias) * units = summary Param 갯수 (Dense모델)

model.add(Conv2D(filters=64, kernel_size=(3, 3)
                 ,padding= 'same' ,
                 input_shape=(28, 28, 1)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (2,2), 
                 padding="valid",  # 디폴트 값
                 activation="relu")) 
model.add(Flatten())  # (N, )
model.add(Dense(32, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(10, activation="softmax"))

model.summary()
# (kernel_size * channels + bias) + filters  = summary Param 갯수 (CNN모델)



