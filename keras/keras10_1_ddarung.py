#데이콘 따릉이 문제풀이
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#1. 데이터
path = './_data/ddarung/'
train_set = pd.read_csv(path + 'train.csv', 
                        index_col=0) # 0번째 컬럼은 인덱스
# print(train_set)
# print(train_set.shape) # (1459, 10)

test_set = pd.read_csv(path + 'test.csv', # 예측에서 쓸거야 !!!
                       index_col=0)
# print(test_set)
# print(test_set.shape) # (715, 9)

# print(train_set.columns) 
# print(train_set.info()) #각 컬럼에 대한 디테일한 내용 (null=중간에 빠진값=결측치)
# print(train_set.describe())

#### 결측치 처리 1. 제거 ####
#print(train_set.isnull().sum())  #null의 컬럼당 개수를 확인. 
train_set =  train_set.dropna()
#print(train_set.isnull().sum()) 
#print(train_set.shape)           # (1328, 10)
###################
test_set = test_set.fillna(test_set.mean())
#replace(to_replace=np.nan, value=0)

x = train_set.drop(['count'], axis=1) #drop 지운다. axis 열을 따라 동작함.
#print(x)
#print(x.columns)
#print(x.shape) # (1459, 9)

y = train_set['count']
# print(y)
# print(y.shape) # (1459, )


x_train, x_test, y_train, y_test = train_test_split(x,y,
             train_size=0.75, shuffle=True, random_state=777)
#2. 모델구성
model = Sequential()
model.add(Dense(100,input_dim=9))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=800, batch_size=100)

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict) : # def 함수를 만드는거 
    return np. sqrt(mean_squared_error(y_test, y_predict)) # 루트 씌움

rmse = RMSE(y_test, y_predict) 
print("RMSE : ", rmse)

y_summit = model.predict(test_set)

#print(y_summit)
#print(y_summit.shape) # (715, 1)

############## .to_csv() 를 사용해서
##### subnission.csv를 완성하시오.


submission = pd.read_csv('C:\study\_data\ddarung\submission.csv',index_col=0)
submission['count'] = y_summit
submission.to_csv('C:\study\_data\ddarung\submission.csv', index=True)

# loss :  38.345489501953125
# RMSE :  53.93673848530674







