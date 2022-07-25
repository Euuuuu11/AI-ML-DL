import numpy as np
from pandas import Categorical
from pyrsistent import m
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from tensorflow.python.keras.models import Sequential, Model   
from tensorflow.python.keras.layers import Dense, Input, Dropout,Conv1D, Flatten
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

# import tensorflow as tf
# tf.random.set_seed(15)

#1. 데이터
datasets = load_iris()
# print(datasets.DESCR) 
# print(datasets.feature_names) # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'] 
x = datasets['data']
y = datasets.target
# print(x)
# print(y)
# print(x.shape, y.shape) # (150, 4) (150, )
print("y의 라벨값 : ", np.unique(y)) # y의 라벨값 : [0 1 2]

# ValueError: in user code:
#     ValueError: Shapes (1, 1) and (1, 3) are incompatible
# y.shape를 (150,1) -> (150,3) 으로 바꿔줘야 해결이 된다.
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

# print(y)
# print(y.shape) # (150, 3)


x_train, x_test, y_train, y_test = train_test_split(x,y,
             train_size=0.8, shuffle=True, random_state=66)


scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
#print(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape,x_test.shape)   #  (120, 4) (30, 4)
x_train = x_train.reshape(120, 2, 2)
x_test = x_test.reshape(30, 2, 2)

#2. 모델구성
model = Sequential()
model.add(Conv1D(64, 2, input_shape=(2,2)))
model.add(Flatten())
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(3, activation = 'relu'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy']) # 다중분류에는 loss = 'categorical_crossentropy'만 사용 

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
date= datetime.datetime.now()      # 2022-07-07 17:22:07.702644
date = date.strftime("%m%d_%H%M")  # 0707_1723
print(date)

filepath = './_ModelCheckpoint/k26_5/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

import time


from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=400, mode='min', 
              verbose=1, restore_best_weights=True) 
mcp = ModelCheckpoint(monitor='val_loss', mode='auto',verbose=1,
                      save_best_only=True,filepath= "".join([filepath,'k26_',date, '_', filename]))
start_time = time.time()
model.fit(x_train, y_train,
          epochs=500, batch_size=100,validation_split=0.2,
          verbose=1, callbacks=[es])

end_time = time.time() - start_time
#4. 평가, 예측
result = model.evaluate(x_test,y_test)
print("loss : ", result[0])
print("accuracy : ", result[1])


from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
# print(y_predict)
y_test = np.argmax(y_test, axis=1)
# print(y_test)

acc = accuracy_score(y_test, y_predict)
print('acc스코어 : ', acc)
print('끝난시간 : ', end_time)


# dropout 적용 후 
# loss :  0.9333333373069763
# acc스코어 :  0.9333333333333333

# loss :  0.19042113423347473
# acc스코어 :  0.9666666666666667
# 끝난시간 :  15.59496521949768