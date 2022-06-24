import numpy as np  
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1.데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
            [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4]]             
            )
y = np.array([11,12,13,14,15,16,17,18,19,20])
print(x.shape) #(2, 10)
print(y.shape) #(10, )
x = x.T
# x = x.transpose()
# x = x.reshape(10,2)    
print(x)
print(x.shape)    #10, 2)

#2. 모델구성
model = Sequential()
model.add(Dense(5,input_dim=2))
model.add(Dense(1))
model.add(Dense(10))
model.add(Dense(6))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x, y, epochs=110, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss :', loss)
result = model.predict([[10, 1.4]])
print('[10,14]의 예측값 :', result)


# loss : 0.7053157091140747
# [10,14]의 예측값 : [[20.123161]]