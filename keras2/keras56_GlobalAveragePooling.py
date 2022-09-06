import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout
import keras
from tensorflow.keras.layers import GlobalAveragePooling2D
#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.

# from tensorflow.keras.utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

#2. 모델
drop = 0.2
activation = 'relu'
optimizer = 'adam'

inputs = Input(shape=(28, 28, 1), name = 'input')
x = Conv2D(128, (2, 2), padding = 'valid',
           activation=activation, name = 'hidden1')(inputs)  # 27, 27, 128
x = Dropout(drop)(x)
x = Conv2D(64, (2, 2), padding = 'same',
           activation=activation, name = 'hidden2')(x)       # 27, 27, 64
x = Dropout(drop)(x)
x = MaxPooling2D()(x)
x = Conv2D(32, (3, 3), padding = 'valid',
           activation=activation, name = 'hidden3')(x)       # 13, 13, 64
x = Dropout(drop)(x)
# x = Flatten()(x)                                            # 12 * 12 * 32 = 387300
# 차원을 바꿔서 쭉 펴주는 건 좋지만, node 갯수가 너무 많이 늘어나는 단점이 있다 .  
x = GlobalAveragePooling2D()(x)

x = Dense(100, activation=activation, name = 'hidden4')(x)   #  
x = Dropout(drop)(x)
outputs = Dense(10, activation = 'softmax', name = 'outputs')(x)

model = Model(inputs=inputs, outputs=outputs)

# model.summary()


#3. 컴파일, 훈련
model.compile(optimizer=optimizer, metrics=['acc'],
                loss='sparse_categorical_crossentropy')


import time
start = time.time()
model.fit(x_train, y_train, epochs=5, validation_split=0.4,
          batch_size=128)
end = time.time()

loss, acc = model.evaluate(x_test, y_test)

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)

print(y_predict[:10])
y_predict = np.argmax(model.predict(x_test), axis=-1)

print("걸린시간 : ", end - start)
print('accuracy_score : ', accuracy_score(y_test, y_predict))

# accuracy_score :  0.9521

