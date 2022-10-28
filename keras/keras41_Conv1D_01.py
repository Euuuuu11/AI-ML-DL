import numpy as np
from  tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, SimpleRNN, Dropout,LSTM,Conv1D, Flatten



#1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.array([[1, 2, 3],[2, 3, 4],[3, 4, 5],[4, 5, 6],[5, 6, 7],[6, 7, 8],[7, 8, 9]])
y = np.array([4,5,6,7,8,9,10])

# print(x.shape, y.shape)   # (7, 3) (7, )
# x의_shape = (행, 열, 몇개씩 자르는지)
x = x.reshape(7, 3, 1)
# print(x.shape)  # (7, 3, 1) 

#2. 모델구성   
model = Sequential()
model.add(SimpleRNN(32, input_shape=(3, 1),return_sequences=True))
# model.add(Conv1D(10, 2, input_shape=(3,1)))
model.add(Flatten())
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(1))
model.summary()  # LSTM : 517 // Conv1D : 97


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=256, mode='auto', verbose=1, 
                              restore_best_weights=True) 

model.fit(x, y, epochs=1007, batch_size=32, callbacks=[es])

#4. 평가, 예측
loss = model.evaluate(x,y)
y_pred = np.array([8,9,10]).reshape(1, 3, 1)    # [[[8], [9], [10]]]
result = model.predict(y_pred)
print('loss : ', loss)
print('[8,9,10]의 결과 : ', result)
model.summary() 

# loss :  9.56846124609001e-05
# [8,9,10]의 결과 :  [[10.682619]]

# loss :  0.061515722423791885
# [8,9,10]의 결과 :  [[10.857255]]