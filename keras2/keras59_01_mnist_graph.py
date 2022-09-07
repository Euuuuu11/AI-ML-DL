from tabnanny import verbose
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout, GlobalAveragePooling2D
import keras
import time

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.

# from tensorflow.keras.utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

# 2. 모델구성

activation = 'relu'
drop = 0.2
optimizer = 'adam'

inputs = Input(shape=(28,28,1), name='input')
x = Conv2D(128, (2, 2), activation=activation, padding='valid', name='hidden1')(inputs)
x = Dropout(drop)(x)
x = MaxPooling2D()(x)
x = Conv2D(32, (3, 3), activation=activation, padding='valid', name='hidden3')(x)
x = Dropout(drop)(x) 
# x = Flatten()(x) # (25*25*32) / Flatten의 문제점: 연산량이 너무 많아짐
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation=activation, name='hidden4')(x)
x = Dropout(drop)(x)
x = Dense(128, activation=activation, name='hidden5')(x)
x = Dropout(drop)(x)
outputs = Dense(10, activation='softmax', name='outputs')(x)

model = Model(inputs=inputs, outputs=outputs)

#3. 컴파일, 훈련
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', patience=15, mode='min', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=7, mode='auto', verbose=1,
                              factor=0.5)
from keras.optimizers import Adam
learning_rate = 0.01
optimizer = Adam(learning_rate=learning_rate)

model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer,
             metrics=['acc'])

import time
start = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1,
                 callbacks=[es, reduce_lr], validation_split=0.2)
end = time.time() - start

loss, acc = model.evaluate(x_test, y_test)
print('larning_rate : ', learning_rate)
print('loss : ', round(loss, 4))
print('acc : ', round(acc, 4))
print('걸린시간 : ', round(end, 4))


################## 시각화 ##################
import matplotlib.pyplot as plt
plt.figure(figsize=(9, 5))

#1. 
plt.subplot(2, 1, 1)
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')

#2. 
plt.subplot(2, 1, 1)
plt.plot(hist.history['acc'], marker='.', c='red', label='acc')
plt.plot(hist.history['val_acc'], marker='.', c='blue', label='val_acc')
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epochs')
plt.legend(['acc', 'val_acc'])

plt.show()

