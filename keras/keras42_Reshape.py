# 만들기
# acc 0.98 이상

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Conv1D, LSTM, Reshape   
from tensorflow.keras.datasets import mnist,fashion_mnist, cifar10, cifar100
import numpy as np


#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000, )
# print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000, )

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
# # print(x_train.shape)             # (60000, 28, 28, 1)
# print(np.unique(y_train, return_counts=True))

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# print(np.unique(y_train, return_counts=True))
# print(np.unique(y_test, return_counts=True))
# print(y_train, y_test)
# print(y_test.shape, y_train.shape)


#2. 모델구성
model  = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3, 3)
                 ,padding= 'same' ,
                 input_shape=(28, 28, 1)))
model.add(MaxPooling2D())                   # (N, 14, 14, 64)
model.add(Conv2D(32, (3,3)))                # (N, 12, 12, 32) 
model.add(Conv2D(7, (3,3)))                 # (N, 10, 10, 7)
model.add(Flatten())                        # (N, 700 )
model.add(Dense(100, activation="relu"))    # (N, 100)
model.add(Reshape(target_shape=(100,1)))    # (N, 100, 1)       # Reshape은 연산하지 않는다.
model.add(Conv1D(10, 3))                    # (N, 98, 10)
model.add(LSTM(16))                         # (N, 16)
model.add(Dense(32, activation="relu"))     # (N, 32)
model.add(Dense(10, activation="softmax"))  # (N, 10)

model.summary()
# (kernel_size * channels + bias) + filters  = summary Param 갯수 (CNN모델)

# #3. 컴파일, 훈련
# model.compile(loss='categorical_crossentropy', optimizer='adam',
#               metrics=['accuracy'])   # 이진분류에 한해 로스함수는 무조건 99퍼센트로 'binary_crossentropy'
#                                       # 컴파일에있는 metrics는 평가지표라고도 읽힘
                                      
# from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
# import datetime
# date= datetime.datetime.now()      # 2022-07-07 17:22:07.702644
# date = date.strftime("%m%d_%H%M")  # 0707_1723
# print(date)

# filepath = './_ModelCheckpoint/k28_2/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

# es = EarlyStopping(monitor='val_loss', patience=60, mode='auto', verbose=1, 
#                               restore_best_weights=True)        
# mcp = ModelCheckpoint(monitor='val_loss', mode='auto',verbose=1,
#                       save_best_only=True,filepath= "".join([filepath,'k28_',date, '_', filename])) 
        


# model.fit(x_train, y_train, epochs=100, batch_size=100,
#                  validation_split=0.2,
#                  callbacks=[es],
#                  verbose=1)

# #4. 평가,예측
# result = model.evaluate(x_test,y_test)
# print("loss : ", result[0])
# print("accuracy : ", result[1])

# from sklearn.metrics import accuracy_score
# y_predict = model.predict(x_test)
# y_predict = np.argmax(y_predict, axis=1)
# y_test = np.argmax(y_test, axis=1)

# acc = accuracy_score(y_test, y_predict)
# print('acc스코어 : ', acc)

# # [과제]
# # loss :  0.06705739349126816
# # acc스코어 :  0.9824