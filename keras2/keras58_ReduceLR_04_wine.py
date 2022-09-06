import numpy as np
from sklearn.datasets import load_iris, load_wine
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Activation, Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout, MaxPool2D, GlobalAveragePooling2D
import time
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target

print(np.unique(y)) # [0 1 2]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66, shuffle=True, stratify=y)


# from keras.utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

#2. 모델
from tensorflow.python.keras.optimizer_v2 import adam, adadelta, adagrad, adamax, rmsprop, nadam

learning_rate = 0.01
optimizer = adam.Adam(lr=learning_rate)
activation = 'relu'
drop = 0.1
inputs = Input(shape=(13,), name='input')
x = Dense(128,
           activation=activation, name='Conv2D1')(inputs)
x = Dropout(drop)(x)
x = Dense(64,
           activation=activation, name='Conv2D2')(x)
x = Dense(32,
           activation=activation, name='Conv2D3')(x)
x = Dense(16, activation=activation, name='hidden3')(x)
outputs = Dense(3, activation='softmax', name='outputs')(x)

model = Model(inputs=inputs, outputs=outputs)
model.summary()

#3. 컴파일, 훈련
model.compile(optimizer=optimizer, metrics=['acc'], loss='sparse_categorical_crossentropy')

from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau

earlyStopping = EarlyStopping(monitor='val_loss', patience=140, mode='auto', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=70, factor=0.5, model = 'auto', verbose=1) # ReduceLROnPlateau: 학습률을 조정해주는 함수

start = time.time()
model.fit(x_train, y_train, epochs=500, validation_split=0.2, batch_size=200, verbose=1, callbacks=[reduce_lr, earlyStopping])
end = time.time() - start
result = model.evaluate(x_test, y_test)
print('걸린시간 : ', end)

print('result : ', result)

from sklearn.metrics import accuracy_score
y_pred = model.predict(x_test)
print('accuracy_score : ', accuracy_score(y_test, np.argmax(y_pred, axis=1)))

# 걸린시간 :  5.515724182128906
# result :  [1.0903899669647217, 0.3888888955116272]
# accuracy_score :  0.3888888888888889