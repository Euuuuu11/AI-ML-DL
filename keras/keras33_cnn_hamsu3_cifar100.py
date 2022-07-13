from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D,Dropout, Input   
from tensorflow.keras.datasets import mnist,fashion_mnist, cifar10, cifar100
import numpy as np


#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)   # (10000, 32, 32, 3) (10000, 1)

# x_train = x_train.reshape(60000, 28, 28, 1)
# x_test = x_test.reshape(10000, 28, 28, 1)
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
input1 = Input(shape=(32, 32, 3)) 
Max1 = (MaxPooling2D(2)(input1))  
drp1 = (Dropout(0.25))(Max1)     
Con1 = Conv2D(64, 3)(drp1)                
Max2 = (MaxPooling2D(2, 2))(Con1)   
drp2 = (Dropout(0.25))(Max2)     
Con2 = Conv2D(128, (3, 3), padding='same', activation='relu')(drp2)
Max3 = (MaxPooling2D(2, 2))(Con2)     
drp3 = (Dropout(0.4))(Max3)     
Con3 = Conv2D(128, (3, 3), padding='same', activation='relu')(drp3)
Max4 = (MaxPooling2D(2, 2))(Con3)     
drp4 = (Dropout(0.4))(Max4)
Flt1 =(Flatten())(drp4)   
dense1 = Dense(128, activation='tanh')(Flt1)
drp5 = (Dropout(0.2))(dense1)
output1 = (Dense(100, activation='softmax'))(drp5)
model = Model(inputs=input1, outputs=output1)
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

filepath = './_ModelCheckpoint/k33_3/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

es = EarlyStopping(monitor='val_loss', patience=60, mode='auto', verbose=1, 
                              restore_best_weights=True)        
mcp = ModelCheckpoint(monitor='val_loss', mode='auto',verbose=1,
                      save_best_only=True,filepath= "".join([filepath,'k33_',date, '_', filename])) 
        


model.fit(x_train, y_train, epochs=100, batch_size=128,
                 validation_split=0.3,
                 callbacks=[es,mcp],
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