from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D,Dropout, LSTM   
from tensorflow.keras.datasets import mnist,fashion_mnist, cifar10, cifar100
import numpy as np


#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)   # (10000, 32, 32, 3) (10000, 1)

x_train = x_train.reshape(50000, 32, 96)
x_test = x_test.reshape(10000, 32, 96)
# print(x_train.shape)             # (50000, 32, 32, 3)
# print(np.unique(y_train, return_counts=True))

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(np.unique(y_train, return_counts=True))
print(np.unique(y_test, return_counts=True))
print(y_train, y_test)
print(y_test.shape, y_train.shape)


#2. 모델구성
model = Sequential()
model.add(LSTM(100, return_sequences=True, activation= 'relu' ,input_shape = (32,96)))    
model.add(LSTM(90, return_sequences=False, activation = 'relu'))  
model.add(Dense(128, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='softmax'))
 

# model.summary()
# (kernel_size * channels + bias) + filters  = summary Param 갯수 (CNN모델)

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='Nadam',
              metrics=['accuracy'])   # 이진분류에 한해 로스함수는 무조건 99퍼센트로 'binary_crossentropy'
                                      # 컴파일에있는 metrics는 평가지표라고도 읽힘
                                      
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
# date= datetime.datetime.now()      # 2022-07-07 17:22:07.702644
# date = date.strftime("%m%d_%H%M")  # 0707_1723
# print(date)

# filepath = './_ModelCheckpoint/k29_2/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

es = EarlyStopping(monitor='val_loss', patience=60, mode='auto', verbose=1, 
                              restore_best_weights=True)        
# mcp = ModelCheckpoint(monitor='val_loss', mode='auto',verbose=1,
                    #   save_best_only=True,filepath= "".join([filepath,'k29_',date, '_', filename])) 
        


model.fit(x_train, y_train, epochs=60, batch_size=512,
                 validation_split=0.3,
                 callbacks=[es],
                 verbose=1)

#4. 평가,예측
result = model.evaluate(x_test,y_test)
print("loss : ", result[0])
print("accuracy : ", result[1])

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test, axis=1)

acc = accuracy_score(y_test, y_predict)
print('acc스코어 : ', acc)


# loss :  3.4802539348602295
# acc스코어 :  0.1779
