from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout,Conv1D, Flatten
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

print(x_train.shape,x_test.shape)   #  (353, 10) (89, 10)
x_train = x_train.reshape(353, 5, 2)
x_test = x_test.reshape(89, 5, 2)

#2. 모델구성
model = Sequential()
model.add(Conv1D(32, 2, input_shape=(5,2)))
model.add(Flatten())
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(1, activation = 'relu'))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam',metrics=['mae'])

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
date= datetime.datetime.now()      # 2022-07-07 17:22:07.702644
date = date.strftime("%m%d_%H%M")  # 0707_1723
print(date)

filepath = './_ModelCheckpoint/k26_3/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
import time

from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=400, mode='min', 
              verbose=1, restore_best_weights=True) 
mcp = ModelCheckpoint(monitor='val_loss', mode='auto',verbose=1,
                      save_best_only=True,filepath= "".join([filepath,'k26_',date, '_', filename]))
start_time = time.time()
model.fit(x_train, y_train, epochs=100, validation_split=0.2,
                 batch_size=15, verbose=1, callbacks=[es])
end_time = time.time() - start_time
y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('끝난시간 : ', end_time)
print('r2스코어 : ', r2)




# dropout 적용 후 
# loss :  3290.4560546875
# r2스코어 :  0.5016890224915627

# loss :  2290.364013671875
# r2스코어 :  0.6531442590273249
# 끝난시간 :  10.049986839294434