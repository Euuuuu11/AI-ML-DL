import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,3,5,4,7,6,7,11,9,7])

#2. 모델
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

model = Sequential()
model.add(Dense(1000, input_dim=1))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1))

#3. 컴파일, 훈련
# from tensorflow.python.keras.optimizer_v1 import Adam, Adadelta, Adagrad, Adamax
# from tensorflow.python.keras.optimizer_v1 import RMSprop, SGD, Nadam
from tensorflow.python.keras.optimizer_v2 import  adam, adadelta, adagrad, adamax
from tensorflow.python.keras.optimizer_v2 import  rmsprop, nadam
# from keras.optimizers import Adam, Adadelta, Adagrad, Adamax
# from keras.optimizers import RMSprop, SGD, Nadam

learning_rate = 0.0001
# optimizer = adam.Adam(learning_rate=learning_rate)
# optimizer = adadelta.Adadelta(learning_rate=learning_rate)
# optimizer = adagrad.Adagrad(learning_rate=learning_rate)
# optimizer = adamax.Adamax(learning_rate=learning_rate)
optimizer = rmsprop.RMSProp(learning_rate=learning_rate)
# optimizer = nadam.Nadam(learning_rate=learning_rate)\
model.compile(loss='mse', optimizer=optimizer)

model.fit(x, y, epochs=50, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y)
y_predict = model.predict([11])

print('loss : ', round(loss, 4), 'lr : ', learning_rate, '결과물 : ', y_predict)
# loss :  2.931 lr :  0.0001 결과물 :  [[9.608019]]
# loss :  36.7842 lr :  0.0001 결과물 :  [[0.9814846]]
# loss :  2.5876 lr :  0.0001 결과물 :  [[11.1229315]]
# loss :  2.5479 lr :  0.0001 결과물 :  [[10.746758]]
# loss :  2.5325 lr :  0.0001 결과물 :  [[11.765221]]
# loss :  2.3677 lr :  0.0001 결과물 :  [[10.604613]]

# optimizers = [adam.Adam, adadelta.Adadelta, adagrad.Adagrad, adamax.Adamax, rmsprop.RMSProp, nadam.Nadam]

# for i in optimizers