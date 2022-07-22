from keras.datasets import reuters
import numpy as np
import pandas as pd

(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words=10000, test_split=0.2
) 

print(x_train)
print(x_train.shape, x_test.shape)    # (8982,) (2246,)
print(y_train)
print(np.unique(y_train,return_counts=True)) # 46   
print(len(np.unique(y_train)))  # 46

print(type(x_train),type(y_train))  # <class 'numpy.ndarray'> <class 'numpy.ndarray'>
print(type(x_train[0]))             # <class 'list'>
# print(x_train[0].shape)           # AttributeError: 'list' object has no attribute 'shape'
print(len(x_train[0]))              # 87    
print(len(x_train[1]))              # 56

print('뉴스기사의 최대길이 :', max(len(i) for i in x_train))    #  2376    
print('뉴스기사의 평균길이 : ', (sum(map(len,x_train))/ len(x_train)))  # 145.5398574927633

#전처리
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

x_train = pad_sequences(x_train, padding='pre', maxlen=100, truncating='pre')
                            # (8982, ) -> (8982, 100)    
x_test = pad_sequences(x_test, padding='pre', maxlen=100, truncating='pre')
    
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape, y_train.shape) # (8982, 100) (8982, 46)
print(x_test.shape, y_test.shape)   # (2246, 100) (2246, 46)

# 시작!!
#2. 모델 구성
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Embedding,Conv1D, Flatten
model = Sequential()     
model.add(Embedding(10000,20,input_length=100))
model.add(LSTM(32))
model.add(Dense(8,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(46,activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_test, y_test, epochs=20, batch_size=128)

#4. 평가, 예측
acc = model.evaluate(x_test, y_test)[1]
print('acc : ', acc)


# acc :  0.6251112818717957








 