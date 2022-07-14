import numpy as np
from sklearn.datasets import fetch_covtype
from tensorflow.python.keras.models import Sequential   
from tensorflow.python.keras.layers import Dense, Dropout,Conv1D, Flatten
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler


#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target
print(x.shape,y.shape) # (581012, 54) (581012,)
print(np.unique(y, return_counts=True))    # [1 2 3 4 5 6 7]
# (array([1, 2, 3, 4, 5, 6, 7]),
# array([211840, 283301,  35754,   2747,   9493,  17367,  20510],
# dtype=int64))
#1. tensorflow의 to_categorical
# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y) #categorical은 앞에 0부터 시작 그래서 8로 나옴

#2. pandas의 get_dummies
y=pd.get_dummies(y)
# print(y.shape)  # (581012, 7)
x_train, x_test, y_train, y_test = train_test_split(x,y,
             train_size=0.8, shuffle=True, random_state=66)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
print(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape,x_test.shape)   #  (464809, 54) (116203, 54)
x_train = x_train.reshape(464809, 9, 6)
x_test = x_test.reshape(116203, 9, 6)

#2. 모델구성
model = Sequential()
model.add(Conv1D(64, 2, input_shape=(9,6)))
model.add(Flatten())
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(7, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
date= datetime.datetime.now()      # 2022-07-07 17:22:07.702644
date = date.strftime("%m%d_%H%M")  # 0707_1723
print(date)

filepath = './_ModelCheckpoint/k26_8/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

import time


from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=100, mode='min', 
              verbose=1, restore_best_weights=True) 
mcp = ModelCheckpoint(monitor='val_loss', mode='auto',verbose=1,
                      save_best_only=True,filepath= "".join([filepath,'k26_',date, '_', filename]))
start_time = time.time()
model.fit(x_train, y_train,
          epochs=15, batch_size=100,validation_split=0.2,
          verbose=1, callbacks=[es])

end_time = time.time() - start_time
#4. 평가,예측
result = model.evaluate(x_test,y_test)
print("loss : ", result[0])
print("accuracy : ", result[1])

print("============================================") 
from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
y_predict = tf.argmax(y_predict,axis=1)
y_test = tf.argmax(y_test,axis=1)

# from sklearn.metrics import accuracy_score
# y_predict = model.predict(x_test)
# # print(y_predict.shape)
# y_predict = np.argmax(y_predict, axis=1)
# y_test = np.argmax(y_test, axis=1)

acc = accuracy_score(y_test, y_predict)
print('acc스코어 : ', acc)
print('끝난시간 : ', end_time)
# dropout 적용 후 
# loss :  0.3710891008377075
# acc스코어 :  0.8374654699104154

# loss :  0.6415029168128967
# acc스코어 :  0.7213755238677143
# 끝난시간 :  200.02193307876587

#21212