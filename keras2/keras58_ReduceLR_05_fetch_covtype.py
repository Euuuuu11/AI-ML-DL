from dataclasses import dataclass
from multiprocessing.dummy import active_children
import numpy as np
from sklearn.datasets import load_iris, load_wine, fetch_covtype
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Activation, Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout, MaxPool2D, GlobalAveragePooling2D
import time
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target

print(np.unique(y)) # [0 1 2]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66, shuffle=True, stratify=y)


# from keras.utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

#2. 모델
from tensorflow.python.keras.optimizer_v2 import adam, adadelta, adagrad, adamax, rmsprop, nadam

learning_rate = 0.001
optimizer = adam.Adam(lr=learning_rate)
activation = 'relu'
drop = 0.2
inputs = Input(shape=(54,), name='input')
x = Dense(30, name='hidden2')(inputs)
x = Dense(20,
           activation=activation, name='Conv2D1')(inputs)
x = Dense(15,
           activation='sigmoid', name='Conv2D2')(x)
x = Dense(20,
           activation=activation, name='Conv2D3')(x)
x = Dense(20,
           activation=activation, name='Conv2D4')(x)

outputs = Dense(7, activation='softmax', name='outputs')(x)

model = Model(inputs=inputs, outputs=outputs)

#3. 컴파일, 훈련
model.compile(optimizer=optimizer, metrics=['acc'], loss='sparse_categorical_crossentropy')

from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau

earlyStopping = EarlyStopping(monitor='val_loss', patience=50, mode='auto', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=20, factor=0.5, model = 'auto', verbose=1) # ReduceLROnPlateau: 학습률을 조정해주는 함수

start = time.time()
model.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=500, verbose=1, callbacks=[reduce_lr, earlyStopping])
end = time.time() - start
result = model.evaluate(x_test, y_test)
print('걸린시간 : ', end)

print('result : ', result)

from sklearn.metrics import accuracy_score
y_pred = model.predict(x_test)
print('accuracy_score : ', accuracy_score(y_test, np.argmax(y_pred, axis=1)))
# 걸린시간 :  15.11889123916626
# result :  [0.16043372452259064, 0.9444444179534912]
# accuracy_score :  0.9444444444444444