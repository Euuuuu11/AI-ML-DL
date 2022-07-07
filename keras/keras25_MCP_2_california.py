from sklearn import datasets
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np  
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
             train_size=0.8, random_state=72)


# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler = MaxAbsScaler()
# scaler = RobustScaler()

# scaler.fit(x_train)
# #print(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# print(x.shape, y.shape)

print(datasets.feature_names)
print(datasets.DESCR)

#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=8,activation='relu'))
model.add(Dense(80))
model.add(Dense(80,activation='relu'))
model.add(Dense(60,activation='relu'))
model.add(Dense(60,activation='relu'))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam',metrics=['mae'])

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
date= datetime.datetime.now()      # 2022-07-07 17:22:07.702644
date = date.strftime("%m%d_%H%M")  # 0707_1723
print(date)

filepath = './_ModelCheckpoint/k25_2/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

es = EarlyStopping(monitor='val_loss', patience=100, mode='min', 
              verbose=1, restore_best_weights=True) 
mcp = ModelCheckpoint(monitor='val_loss', mode='auto',verbose=1,
                      save_best_only=True,filepath= "".join([filepath,'k25_',date, '_', filename]))

model.fit(x_train, y_train, epochs=1000, validation_split=0.2,
                 batch_size=105, verbose=1, callbacks=[es,mcp])

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)




#1. 스케일러 하기 전
#  loss :  0.4144200384616852
#  r2스코어 :   0.6836346755251912

#2. MinMaxScaler 
# loss :  0.4348423480987549
# r2스코어 :  0.6680446118395076

#3. StandardScaler  
# loss :  0.425083190202713
# r2스코어 :   0.6754946968248441

#4. MaxAbsScaler 
# loss :  0.42158570885658264
# r2스코어 :  0.6781645820794922

#5. RobustScaler 
#  loss :  0.2656779885292053
#  r2스코어 :   0.7971834530040288