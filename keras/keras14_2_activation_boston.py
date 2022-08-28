# [과제]
# activation : sigmoid, relu, linear 넣어라
# metrics 추가
# EarlyStopping 넣고 성능비교
# 감상문 2줄 이상!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

from subprocess import call
from tabnanny import verbose
from sklearn.datasets import load_boston
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np  
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets. target

x_train, x_test, y_train, y_test = train_test_split(x,y,
             train_size=0.8, shuffle=True, random_state=66)
# print(x)
# print(y)
# print(x.shape, y.shape)  # (506, 13) (506, )

# print(datasets.feature_names) 
# print(datasets.DESCR)


#2. 모델구성
model = Sequential()
model.add(Dense(128, input_dim=13, activation='relu'))
model.add(Dense(64))
model.add(Dense(64, activation='relu'))
model.add(Dense(32))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

import time
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam',metrics=['mae'])

from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=100, mode='min', 
              verbose=1, restore_best_weights=True) 

start_time = time.time()
print(start_time)

model.fit(x_train, y_train,
          epochs=1000, batch_size=1,validation_split=0.2,
          verbose=1, callbacks=[es] )

end_time = time.time() - start_time
print("걸린시간 : ", end_time)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss) 

#1. validation 적용
# loss :  14.096229553222656
# r2스코어 :  0.8293788022088331


#2. EarlyStopping 적용
# loss :  13.131475448608398
# r2스코어 :  0.8428928454653812

#3. activation 적용
# loss :  [9.31887435913086, 2.3453245162963867]
# r2스코어 :  0.8885074606958558

