#데이콘 따릉이 문제풀이
import numpy as np
import pandas as pd
from sklearn import metrics
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

#1. 데이터
path = './_data/ddarung/'
train_set = pd.read_csv(path + 'train.csv', 
                        index_col=0) 


test_set = pd.read_csv(path + 'test.csv', 
                       index_col=0)



train_set =  train_set.dropna()

test_set = test_set.fillna(test_set.mean())


x = train_set.drop(['count'], axis=1) 
#

y = train_set['count']



x_train, x_test, y_train, y_test = train_test_split(x,y,
             train_size=0.75, shuffle=True, random_state=85)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
print(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
input1 = Input(shape=(9,))
dense1 = Dense(100)(input1)
dense2 = Dense(100,activation='relu')(dense1)
dense3 = Dense(100)(dense2)
output1 = Dense(1)(dense3)
model = Model(inputs = input1, outputs = output1)

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam', metrics=['mse'])

from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=100, mode='min', 
              verbose=1, restore_best_weights=True) 

model.fit(x_train, y_train, epochs=800, batch_size=10, verbose=1, callbacks=[es])


#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)


# loss :  27.7861328125
# r2스코어 :  0.7310630314838201

# 함수모델 변경 후
# loss :  27.437788009643555
# r2스코어 :  0.7469243879504571