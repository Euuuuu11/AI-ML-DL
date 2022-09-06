from dataclasses import dataclass
from multiprocessing.dummy import active_children
import numpy as np
from keras.datasets import mnist, cifar100
from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Activation, Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout, MaxPool2D, GlobalAveragePooling2D
import time
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# import train_test_split
from sklearn.model_selection import train_test_split
print(tf.__version__) # 2.9.1

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66, shuffle=True)


# from keras.utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

#2. 모델
from tensorflow.python.keras.optimizer_v2 import adam, adadelta, adagrad, adamax, rmsprop, nadam

learning_rate = 0.001
optimizer = adam.Adam(lr=learning_rate)
activation = 'relu'
drop = 0.1

inputs = Input(shape=(10,), name='input')
x = Dense(20,
           activation='linear', name='Conv2D1')(inputs)
# x = Dropout(drop)(x)
x = Dense(30,
           activation='linear', name='Conv2D2')(x)
# x = Dropout(drop)(x)
x = Dense(30, activation=activation, name='hidden3')(x)
# x = Dropout(drop)(x)
outputs = Dense(1,activation='linear', name='outputs')(x)

# input1 = Input(shape=(10,))
# dense1 = Dense(200)(input1)
# dense2 = Dense(300)(dense1)
# dense3 = Dense(200)(dense2)
# dense4 = Dense(300, activation='relu')(dense3)
# dense5 = Dense(150)(dense4)
# dense6 = Dense(180)(dense5)
# output1 = Dense(1)(dense6)
# model = Model(inputs=input1, outputs=output1)

model = Model(inputs=inputs, outputs=outputs)
# model.summary()

#3. 컴파일, 훈련
model.compile(optimizer=optimizer, metrics=['mae'], loss='mse')

from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau

earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='auto', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=50, factor=0.5, model = 'auto', verbose=1) # ReduceLROnPlateau: 학습률을 조정해주는 함수

start = time.time()
model.fit(x_train, y_train, epochs=500, validation_split=0.2, batch_size=200, verbose=1, callbacks=[reduce_lr, earlyStopping])
end = time.time() - start
result = model.evaluate(x_test, y_test)
print('걸린시간 : ', end)

print('result : ', result)

from sklearn.metrics import accuracy_score, r2_score
y_pred = model.predict(x_test)
r2_score = r2_score(y_test, y_pred)
print('r2_score : ', r2_score)

# 걸린시간 :  16.48178005218506
# result :  [3312.468505859375, 47.197269439697266]
# r2_score :  0.4896074485278137