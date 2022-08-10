from keras.datasets import mnist
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D  
import numpy as np

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape, y_train.shape)
# print(x_test.shape, y_test.shape) 

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784 )
print(x_train.shape)         

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


#2. 모델구성
model=Sequential()
# model.add(Dense(64,input_shape=(28*28, )))
model.add(Dense(64,input_shape=(784, )))
model.add(Dense(128, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
model.summary()  

# model.summary()
# (kernel_size * channels + bias) + filters  = summary Param 갯수 (CNN모델)

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='Nadam',
              metrics=['accuracy'])   # 이진분류에 한해 로스함수는 무조건 99퍼센트로 'binary_crossentropy'
                                      # 컴파일에있는 metrics는 평가지표라고도 읽힘
                                      
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
date= datetime.datetime.now()      # 2022-07-07 17:22:07.702644
date = date.strftime("%m%d_%H%M")  # 0707_1723
print(date)

filepath = './_ModelCheckpoint/k30_1/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

es = EarlyStopping(monitor='val_loss', patience=60, mode='auto', verbose=1, 
                              restore_best_weights=True)        
mcp = ModelCheckpoint(monitor='val_loss', mode='auto',verbose=1,
                      save_best_only=True,filepath= "".join([filepath,'k30_',date, '_', filename])) 
        

import time
start = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=128,
                 validation_split=0.3,
                 callbacks=[es,mcp],
                 verbose=1)
end = time.time()

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
print("걸린시간 : ", round(end-start, 4))
# loss :  0.2338026911020279
# acc스코어 :  0.9269