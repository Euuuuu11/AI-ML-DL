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
             train_size=0.75, shuffle=True, random_state=31)
#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=9))
model.add(Dense(100, activation='selu'))
model.add(Dense(100, activation='selu'))
model.add(Dense(100, activation='selu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=800, batch_size=10)

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








# loss :  2135.846435546875            epochs=210 -> epochs=400
# RMSE :  46.21521732764418           random_state=777, batch_size=100

# loss :  2124.783447265625            epochs=400 -> epochs=500
# RMSE :  46.09537331966037

# loss :  2118.128662109375           노드량 늘림 .  epochs=250
# RMSE :  46.02313321380063             random_state=750

# loss :  2111.258056640625          다른조건은 다 동일 하고, epochs=888 훈련량 늘림 .
# RMSE :  45.94842612876985           

# loss :  2040.7791748046875               epochs=888 -> epochs=889
# RMSE :  45.17498537358058

# train_size=0.99, batch_size=10 바꾼 후
# loss :  1109.9464111328125           epochs=350
# RMSE :  33.315855532995286

# loss :  904.0668334960938              
# RMSE :  30.06770403590947               epochs=300

# 다른조건은 다 동일하고, mse에서 mae로 바꿔줬을 때 결과값이 mae가 더 좋다.
# loss :  23.577592849731445             
# RMSE :  28.623385344301294              

# loss :  711.4154663085938          train_size=0.999
# RMSE :  26.672367832028506

# loss :  4.9299635887146            train_size=0.9999
# RMSE :  2.2203521728515625

# activation='selu' 사용 후
# loss :  869.8753051757812      batch_size=100 -> batch_size=10 
# RMSE :  29.493652657330966     random_state=777, epochs=350

# loss :  420.6038513183594      위와 동일 .
# RMSE :  20.508630439421065

# activation='relu' 사용 후 # 'relu'라는 함수는 layer에서 
    # layer로 넘어갈 때 음수값이 나오는 nod는 0처리 해준다.                        
# loss :  19.277578353881836
# RMSE :  24.14954499201928

#activation='swish' 사용 후, layer 개수 줄임.
# loss :  22.995586395263672
# RMSE :  26.59749988517869