from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D   
from tensorflow.keras.datasets import mnist,fashion_mnist, cifar10, cifar100
import numpy as np
from sklearn.model_selection import train_test_split



#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000, )
# print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000, )

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
print(x_train.shape)             # (60000, 28, 28, 1)
# print(np.unique(y_train, return_counts=True))

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# print(np.unique(y_train, return_counts=True))
# print(np.unique(y_test, return_counts=True))
# print(y_train, y_test)
# print(y_test.shape, y_train.shape)


#2. 모델구성
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
model.add(Conv2D(8,(2,2),padding='valid', activation='relu'))
model.add(Flatten())  # (N, )
model.add(Dense(32, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(10, activation="softmax"))

#3. 컴파일, 훈련
# from tensorflow.python.keras.optimizer_v1 import Adam, Adadelta, Adagrad, Adamax
# from tensorflow.python.keras.optimizer_v1 import RMSprop, SGD, Nadam
from tensorflow.python.keras.optimizer_v2 import  adam, adadelta, adagrad, adamax
from tensorflow.python.keras.optimizer_v2 import  rmsprop, nadam
# from keras.optimizers import Adam, Adadelta, Adagrad, Adamax
# from keras.optimizers import RMSprop, SGD, Nadam

learning_rate = 0.001
optimizer = adam.Adam(learning_rate=learning_rate)
# optimizer = adadelta.Adadelta(learning_rate=learning_rate)
# optimizer = adagrad.Adagrad(learning_rate=learning_rate)
# optimizer = adamax.Adamax(learning_rate=learning_rate)
# optimizer = rmsprop.RMSProp(learning_rate=learning_rate)
# optimizer = nadam.Nadam(learning_rate=learning_rate)
model.compile(loss='mse', optimizer=optimizer)

model.fit(x_train, y_train, epochs=50, batch_size=128)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict([11])

print('loss : ', round(loss, 4), 'lr : ', learning_rate, '결과물 : ', y_predict)
# loss :  2.931 lr :  0.0001 결과물 :  [[9.608019]]
# loss :  36.7842 lr :  0.0001 결과물 :  [[0.9814846]]
# loss :  2.5876 lr :  0.0001 결과물 :  [[11.1229315]]
# loss :  2.5479 lr :  0.0001 결과물 :  [[10.746758]]
# loss :  2.5325 lr :  0.0001 결과물 :  [[11.765221]]
# loss :  2.3677 lr :  0.0001 결과물 :  [[10.604613]]

# optimizers = [adam.Adam, adadelta.Adadelta, adagrad.Adagrad, adamax.Adamax, rmsprop.RMSProp, nadam.Nadam]

# for i in optimizers