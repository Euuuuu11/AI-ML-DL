import numpy as np
from  tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, SimpleRNN, Dropout, LSTM

#1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.array([[1, 2, 3],[2, 3, 4],[3, 4, 5],[4, 5, 6],[5, 6, 7],[6, 7, 8],[7, 8, 9]])
y = np.array([4,5,6,7,8,9,10])

# print(x.shape, y.shape)   # (7, 3) (7, )
# x의_shape = (행, 열, 몇개씩 자르는지)
x = x.reshape(7, 3, 1)
# print(x.shape)  # (7, 3, 1) 

#2. 모델구성    # LSTM은 반복 모듈은 단순한 한개의 layer가 아닌 4개의 layer가 서로 정보를 주고받는 구조이다.
model= Sequential()                                     # input_dim  
# model.add(SimpleRNN(units=10, input_shape=(3, 1)))    # [batch, timesteps, feature]   # input_shape 행을 뺀다.
# model.add(SimpleRNN(units=10, input_length=3, input_dim=1))
# model.add(SimpleRNN(units=10, input_dim=1, input_length=3))   # 작동은 하나, 가독성이 떨어짐.
model.add(LSTM(units=10, input_shape=(3, 1)))    # [batch, timesteps, feature]   # input_shape 행을 뺀다.
# model.add(SimpleRNN(32))   
# SimpleRNN 여러 번 사용불가
# # ValueError: Input 0 of layer simple_rnn_1 is incompatible with the layer: expected ndim=3, found ndim=2. Full shape received: (None, 64)
model.add(Dense(8, activation='relu')) 
model.add(Dense(4, activation='relu')) 
# model.add(Dropout(0.2))   # 적용하니 성능저하
model.add(Dense(1))

model.summary() 
  
# [LSTM] units : 10 -> 4 * 10 *(1 + 1 + 10) = 480
# 결론 : LSTM = simpleRNN * 4
# 4의 의미는 Cell state, Forget gate, Input gate, Output gate


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=300, mode='auto', verbose=1, 
                              restore_best_weights=True) 

model.fit(x, y, epochs=2022, batch_size=16, callbacks=[es])

#4. 평가, 예측
loss = model.evaluate(x,y)
y_pred = np.array([8,9,10]).reshape(1, 3, 1)    # [[[8], [9], [10]]]
result = model.predict(y_pred)
print('loss : ', loss)
print('[8,9,10]의 결과 : ', result)

# loss :  9.56846124609001e-05
# [8,9,10]의 결과 :  [[10.682619]]  

# LSTM 후           
# loss :  0.0013157735811546445

# [8,9,10]의 결과 :  [[10.752905]]

# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# lstm (LSTM)                  (None, 10)                480
# _________________________________________________________________
# dense (Dense)                (None, 8)                 88
# _________________________________________________________________
# dense_1 (Dense)              (None, 4)                 36
# _________________________________________________________________
# dense_2 (Dense)              (None, 1)                 5
# =================================================================
# Total params: 609
# Trainable params: 609
# Non-trainable params: 0