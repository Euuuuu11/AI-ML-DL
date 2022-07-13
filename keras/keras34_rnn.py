import numpy as np
from  tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, SimpleRNN, Dropout

#1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.array([[1, 2, 3],[2, 3, 4],[3, 4, 5],[4, 5, 6],[5, 6, 7],[6, 7, 8],[7, 8, 9]])
y = np.array([4,5,6,7,8,9,10])

# print(x.shape, y.shape)   # (7, 3) (7, )
# x의_shape = (행, 열, 몇개씩 자르는지)
x = x.reshape(7, 3, 1)
# print(x.shape)  # (7, 3, 1) 

#2. 모델구성    # rnn은 3차원을 2차원으로 보내준다. 장기의존성
model= Sequential()
model.add(SimpleRNN(64, input_shape=(3, 1))) # input_shape 행을 뺀다.
# model.add(SimpleRNN(32))   
# SimpleRNN 여러 번 사용불가
# # ValueError: Input 0 of layer simple_rnn_1 is incompatible with the layer: expected ndim=3, found ndim=2. Full shape received: (None, 64)
model.add(Dense(32, activation='tanh')) 
# model.add(Dropout(0.2))   # 적용하니 성능저하
model.add(Dense(16, activation='relu')) 
model.add(Dense(8, activation='relu')) 
model.add(Dense(1))

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

# loss :  9.56846124609001e-05
# [8,9,10]의 결과 :  [[10.682619]]