import pandas as pd
import numpy as np
from  tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Conv1D, Flatten, MaxPooling2D 

path = './_data\kaggle_jena/'
df  = pd.read_csv(path + 'jena_climate_2009_2016.csv')
# print(df.shape) # (420551, 15)
# df.info()

#데이터 년, 월, 일 구분
df['Date Time'] = pd.to_datetime(df['Date Time'])
df['year'] = df['Date Time'].dt.year  
df['month'] = df['Date Time'].dt.month
df['day'] = df['Date Time'].dt.day
df['hour'] = df['Date Time'].dt.hour
df.drop(['hour','month','day', 'year'], inplace=True, axis=1)
# df.info()
# print(df.shape)   # (420551, 15)

size = 13

def split_x(datasets, size):
    aaa = []
    for i in range(len(datasets) - size + 1):
        subset = datasets[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)
    
bbb = split_x(df, size)
# print(bbb)
# print(bbb.shape)    

x = bbb[:, :-1]
y = bbb[:, -1]
# print(x.shape, y.shape) # (420539, 12, 15) (420539, 15)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
             train_size=0.8, shuffle=True, random_state=66)

print(x_train.shape,x_test.shape)   #(336431, 12, 15) (84108, 12, 15)

#2. 모델구성
model = Sequential()
model.add(Conv1D(64, 2, input_shape=(12,15)))
model.add(Flatten())
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')   
                                      
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=60, mode='auto', verbose=1, 
                              restore_best_weights=True)        

model.fit(x_train, y_train, epochs=1002, batch_size=32,     # 두개 이상은 list이므로, list형식으로 해준다.
                 validation_split=0.2,
                 callbacks=[es],
                 verbose=1)

# #4. 평가,예측
loss = model.evaluate(x_test,y_test)