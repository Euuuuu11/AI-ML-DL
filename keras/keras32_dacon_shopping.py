import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import time
from sklearn.metrics import r2_score, mean_squared_error


#1. 데이터
path = './_data/dacon_shopping/'
train_set = pd.read_csv(path + 'train.csv', # + 명령어는 문자를 앞문자와 더해줌
                        index_col=0) # index_col=n n번째 컬럼을 인덱스로 인식

# print(train_set.shape) # (6255, 12)
# print(train_set.describe())
# print(train_set.columns)

test_set = pd.read_csv(path + 'test.csv', # 예측에서 쓸거임                
                       index_col=0)
# print(test_set.shape) # (180, 11)

train_set = train_set.fillna(0)
test_set = test_set.fillna(0)
# print(test_set)

#데이터 년, 월, 일 구분
train_set['Date'] = pd.to_datetime(train_set['Date'])
train_set['year'] = train_set['Date'].dt.year 
train_set['month'] = train_set['Date'].dt.month
train_set['day'] = train_set['Date'].dt.day
train_set.drop(['Date', 'day', 'year'], inplace=True, axis=1)
train_set['month'] = train_set['month']#.astype('category')

test_set['Date'] = pd.to_datetime(test_set['Date'])
test_set['year'] = test_set['Date'].dt.year 
test_set['month'] = test_set['Date'].dt.month
test_set['day'] = test_set['Date'].dt.day
test_set.drop(['Date', 'day', 'year'], inplace=True, axis=1)
test_set['month'] = test_set['month']#.astype('category')

# 큰 의미 없는 데이터 드랍
train_set = train_set.drop(['Temperature'], axis=1)
test_set = test_set.drop(['Temperature'],axis=1)
train_set = train_set.drop(['Fuel_Price'], axis=1)
test_set = test_set.drop(['Fuel_Price'],axis=1)
train_set = train_set.drop(['Unemployment'], axis=1)
test_set = test_set.drop(['Unemployment'],axis=1)


x = train_set.drop(['Weekly_Sales'], axis=1)
y = train_set['Weekly_Sales']

# print(x.shape)  # (6255, 8)
# print(y.shape)  # (6255, )
# print(x.columns)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                    train_size=0.7,random_state=64)
   


scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
print(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_set = scaler.transform(test_set)





#2. 모델구성
model = Sequential()
model.add(Dense(256, input_dim=8, activation='relu')) 
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))   
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))               
model.add(Dense(1))   
                                           
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam',
              metrics=['mae'])   
                                      
earlyStopping = EarlyStopping(monitor='val_loss', patience=600, mode='auto', verbose=1, 
                              restore_best_weights=True)        

model.fit(x_train, y_train, epochs=1005, batch_size=32,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

def RMSE(y_test, y_predict) :  
    return np. sqrt(mean_squared_error(y_test, y_predict)) 
rmse = RMSE(y_test, y_predict) 
print("RMSE : ", rmse)

y_summit = model.predict(test_set)          

submission = pd.read_csv('C:\study\_data\dacon_shopping\sample_submission.csv',index_col=0)
submission['Weekly_Sales'] = y_summit
submission.to_csv('C:\study\_data\dacon_shopping\sample_submission.csv', index=True)



# MinMaxScaler
# RMSE :  576988.5458324888

# StandardScaler
# RMSE :  547947.5501600921

# MaxAbsScaler
# RMSE :  583482.5175482633

# RobustScaler
# RMSE :  617641.3658200757

# 519824.83852852817