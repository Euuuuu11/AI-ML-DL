import numpy as np
from  tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, SimpleRNN, Dropout, LSTM, GRU

#1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
             [5,6,7],[6,7,8],[7,8,9],[8,9,10],
             [9,10,11],[10,11,12],
             [20,30,40],[30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predict = np.array([50,60,70])

# print(x.shape, y.shape) # (13, 3) (13, )
x = x.reshape(13,3,1)
print(x.shape)

#2. 모델구성    
model = Sequential()
model.add(LSTM(170, return_sequences=True, activation= 'relu' ,input_shape = (3,1)))    # (n,3,1) -> (n,3,170)
model.add(LSTM(90, return_sequences=False, activation = 'relu'))
model.add(Dense(60, activation = 'relu'))
model.add(Dense(20, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(1))
model.summary() 
  
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
# from tensorflow.python.keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor='loss', patience=300, mode='auto', verbose=1, 
#                               restore_best_weights=True) 
model.fit(x, y, epochs=100)

#4. 평가, 예측
loss = model.evaluate(x,y)
y_pred = np.array([60,70,80]).reshape(1, 3, 1)   
x_predict = model.predict(y_pred)
print('loss : ', loss)
print('[60,70,80]의 결과 : ', x_predict)

# loss :  0.0003550819819793105
# [60,70,80]의 결과 :  [[98.13206]]

# loss :  0.00016862916527315974
# [60,70,80]의 결과 :  [[99.732956]]