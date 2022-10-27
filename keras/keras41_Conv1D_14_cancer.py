import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.models import Sequential   
from tensorflow.python.keras.layers import Dense, Dropout,Conv1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

#1. 데이터
datasets = load_breast_cancer()
# print(datasets)
# print(datasets.DESCR) 
# print(datasets.feature_names)

x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(x,y,
             train_size=0.8,random_state=72)
# print(x.shape, y.shape) #(569, 30) (569, )
# print(y)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)
#print(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape,x_test.shape)   #  (455, 30) (114, 30)
x_train = x_train.reshape(455, 15, 2)
x_test = x_test.reshape(114, 15, 2)

# 2. 모델구성
model = Sequential()
model.add(Conv1D(32, 2, input_shape=(15,2)))
model.add(Flatten())
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1, activation = 'sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy']) # 2개 이상은 list

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
date= datetime.datetime.now()      # 2022-07-07 17:22:07.702644
date = date.strftime("%m%d_%H%M")  # 0707_1723
print(date)

filepath = './_ModelCheckpoint/k26_4/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
import time

from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=500, mode='min', 
              verbose=1, restore_best_weights=True) 
mcp = ModelCheckpoint(monitor='val_loss', mode='auto',verbose=1,
                      save_best_only=True,filepath= "".join([filepath,'k26_',date, '_', filename]))

start_time = time.time()
hist = model.fit(x_train, y_train,
          epochs=100, batch_size=10,validation_split=0.2,
          verbose=1,  callbacks=[es] )
end_time = time.time() - start_time
y_predict = model.predict(x_test)
y_predict = y_predict.round(0)
print(y_predict)

##### [과제 1.]accuracy_score 완성
from sklearn.metrics import r2_score, accuracy_score
acc = accuracy_score(y_test, y_predict)

# r2 = r2_score(y_test, y_predict)

print(y_predict)
#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc스코어 : ', acc)
print('끝난시간 : ', end_time)

# dropout 적용 후 
# loss :  0.15960974991321564
# acc스코어 :  0.9649122807017544

# loss :  0.6886007785797119
# acc스코어 :  0.34210526315789475
# 끝난시간 :  19.035841941833496