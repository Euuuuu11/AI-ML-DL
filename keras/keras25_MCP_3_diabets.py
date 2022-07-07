from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
import numpy as np  
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

datasets = load_diabetes()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(x,y,
             train_size=0.8,random_state=72)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)
#print(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# print(x)
# print(y)
# print(x.shape, y.shape) # (442, 10) (442, )

# print(datasets.feature_names)    # ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
# print(datasets.DESCR)   

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=10,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(40,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(60,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam',metrics=['mae'])

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
date= datetime.datetime.now()      # 2022-07-07 17:22:07.702644
date = date.strftime("%m%d_%H%M")  # 0707_1723
print(date)

filepath = './_ModelCheckpoint/k25_3/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=400, mode='min', 
              verbose=1, restore_best_weights=True) 
mcp = ModelCheckpoint(monitor='val_loss', mode='auto',verbose=1,
                      save_best_only=True,filepath= "".join([filepath,'k25_',date, '_', filename]))

model.fit(x_train, y_train, epochs=100, validation_split=0.2,
                 batch_size=15, verbose=1, callbacks=[es,mcp])

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)







#1. 스케일러 하기 전
#  loss :  2356.566650390625
#  r2스코어 :  0.6431184979957294

#2. MinMaxScaler 
# loss :  2261.770751953125
# r2스코어 :  0.6574744812515949

#3. StandardScaler  
# loss :  3004.6962890625
# r2스코어 :   0.5449648980750703

#4. MaxAbsScaler 
# loss :  2317.41845703125
# r2스코어 :  0.6490471057142394

#5. RobustScaler 
#  loss :  2578.9599609375
#  r2스코어 :   0.6094389508664131