from tabnanny import verbose
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
import numpy as np  
from sklearn.preprocessing import OneHotEncoder


datasets = load_diabetes()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(x,y,
             train_size=0.8, shuffle=True, random_state=777)


#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=10,activation='relu'))
model.add(Dense(90))
model.add(Dense(90,activation='relu'))
model.add(Dense(80))
model.add(Dense(80,activation='relu'))
model.add(Dense(40))
model.add(Dense(10,activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam',metrics=['mae'])

from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=100, mode='min', 
              verbose=1, restore_best_weights=True) 

model.fit(x_train, y_train, epochs=1000, validation_split=0.2,
                 batch_size=100, verbose=1, callbacks=[es])

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)


# loss :  2119.338134765625
# r2스코어 :  0.6490527187925481

# validation 사용 후
# loss :  2177.955810546875
# r2스코어 :  0.6701675391497158

# 과제