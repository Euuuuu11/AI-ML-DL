#데이콘 따릉이 문제풀이
import numpy as np
import pandas as pd
from sklearn import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

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
             train_size=0.75, shuffle=True, random_state=31)
#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=9))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam', metrics=['mse'])

from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=100, mode='min', 
              verbose=1, restore_best_weights=True) 

model.fit(x_train, y_train, epochs=800, batch_size=10, verbose=1, callbacks=[es])


#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict) :  
    return np. sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict) 
print("RMSE : ", rmse)



# 과제 











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



