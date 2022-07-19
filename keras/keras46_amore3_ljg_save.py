from more_itertools import zip_equal
import pandas as pd
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, LSTM, Dropout, MaxPooling1D, Conv1D, Flatten, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
import time

path = './_data/test_amore_0718/'
df=pd.read_csv(path + '아모레220718.csv',thousands=',',encoding='cp949')
df1=pd.read_csv(path + '삼성전자220718.csv',thousands=',',encoding='cp949')
# df1.describe()
df = df.sort_values(by='일자', ascending=True) #오름차순 정렬
df1 = df1.sort_values(by='일자', ascending=True) #오름차순 정렬

# 컬럼 삭제
df = df.drop(df.columns[[5]], axis=1)
df1 = df1.drop(df1.columns[[5]], axis=1)

df = df.dropna(axis=0)
df1 = df.dropna(axis=0)
# df = df.fillna(0)
# df1 = df1.fillna(0)

# print(df1.columns)
# Index(['일자', '시가', '고가', '저가', '종가', 'Unnamed: 6', '등락률', '거래량', '금액(백만)',
#        '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비']

df.rename(columns = {'일자':'Date', '시가': 'open', '고가': 'high','저가': 'low','종가':'close','Unnamed: 6':'diff',
                          '등락률':'wk Range','거래량':'volume','금액(백만)':'amount','신용비':'credit cost','개인':'individual',
                          '기관':'agency','외인(수량)':'quantity',
                          '외국계':'foreign','프로그램':'program','외인비':'exogenous'},inplace=True)

df1.rename(columns = {'일자':'Date', '시가': 'open', '고가': 'high','저가': 'low','종가':'close','Unnamed: 6':'diff',
                          '등락률':'wk Range','거래량':'volume','금액(백만)':'amount','신용비':'credit cost','개인':'individual',
                          '기관':'agency','외인(수량)':'quantity',
                          '외국계':'foreign','프로그램':'program','외인비':'exogenous'},inplace=True)
# print(df.columns,df1.columns)
# df.info()

# Date 년, 월, 일, 시간 분리
df['Date'] = pd.to_datetime(df['Date'])
df['year'] = df['Date'].dt.year  
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day
df['hour'] = df['Date'].dt.hour
df.drop(['Date', 'year', 'hour'], inplace=True, axis=1)

df1['Date'] = pd.to_datetime(df1['Date'])
df1['year'] = df1['Date'].dt.year  
df1['month'] = df1['Date'].dt.month
df1['day'] = df1['Date'].dt.day
df1['hour'] = df1['Date'].dt.hour
df1.drop(['Date', 'year', 'hour'], inplace=True, axis=1)

# 필요없는 컬럼 제거
df = df.drop(['amount'], axis = 1)
df1 = df1.drop(['amount'], axis = 1)
df = df.drop(['credit cost'], axis = 1)
df1 = df1.drop(['credit cost'], axis = 1)
df = df.drop(['individual'], axis = 1)
df1 = df1.drop(['individual'], axis = 1)
df = df.drop(['agency'], axis = 1)
df1 = df1.drop(['agency'], axis = 1)
df = df.drop(['quantity'], axis = 1)
df1 = df1.drop(['quantity'], axis = 1)
df = df.drop(['foreign'], axis = 1)
df1 = df1.drop(['foreign'], axis = 1)
df = df.drop(['program'], axis = 1)
df1 = df1.drop(['program'], axis = 1)
df = df.drop(['exogenous'], axis = 1)
df1 = df1.drop(['exogenous'], axis = 1)
 
# 새로운 close2를 만들고, close와 같게 만들어준다.
df['close2'] = df['close']
df1['close2'] = df1['close']

# df.info()
# df1.info()
df = df.drop(['close'], axis = 1)
df1 = df1.drop(['close'], axis = 1)
# df.info()
# df1.info() 
# Dtype 변형
df = df.astype(dtype='float64')
df1 = df1.astype(dtype='float64')
# print(df.shape, df1.shape) # (3170, 9) (3170, 9)

# feature_cols = ['open','high', 'low','volume','diff','wk Range','month','day']
# label_cols = ['close']
df = np.array(df)
df1 = np.array(df1)
print(df.shape, df1.shape) # (3170, 9) (3170, 9)

def split_xy3(dataset, time_steps, y_column) : 
    x, y = list(), list()
    for i in range(len(dataset)) :
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column -1
        if y_end_number > len(dataset) :
            break
        tmp_x = dataset[i:x_end_number, :-1]
        tmp_y = dataset[x_end_number-1:y_end_number, -1]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

size = 3
size2 = 7
        
x1,y1 = split_xy3(df,size2, size)
x2,y2 = split_xy3(df1,size2, size)   

print(x1.shape, y1.shape)  # (3162, 7, 8) (3162, 3)
print(x2.shape, y2.shape)  # (3162, 7, 8) (3162, 3)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scale_cols = ['open', 'high', 'low', 'close2', 'volume','diff','wk Range','month','day']
df_scaled = scaler.fit_transform(df)

df_scaled = pd.DataFrame(df_scaled)
df_scaled.columns = scale_cols
x1_train, x1_test, x2_train, x2_test, y1_train, y1_test = train_test_split(x1, x2, y1, test_size=0.2    , shuffle=False)

# 2. 모델구성
# 2-1. 모델1
input1 = Input(shape=(7, 8))
dense1 = Conv1D(64, 2, activation='relu', name='d1')(input1)
dense2 = LSTM(128, activation='tanh', name='d2')(dense1)
dense3 = Dense(64, activation='relu', name='d3')(dense2)
output1 = Dense(32, activation='relu', name='out_d1')(dense3)

# 2-2. 모델2
input2 = Input(shape=(7, 8))
dense11 = Conv1D(64, 2, activation='relu', name='d11')(input2)
dense12 = LSTM(128, activation='tanh', name='d12')(dense11)
dense13 = Dense(64, activation='relu', name='d13')(dense12)
dense14 = Dense(32, activation='relu', name='d14')(dense13)
output2 = Dense(16, activation='relu', name='out_d2')(dense14)

from tensorflow.python.keras.layers import concatenate
merge1 = concatenate([output1, output2], name='m1')
merge2 = Dense(100, activation='relu', name='mg2')(merge1)
merge3 = Dense(100, name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)

model = Model(inputs=[input1, input2], outputs=[last_output])
import datetime
date = datetime.datetime.now()
print(date)

date = date.strftime("%m%d_%H%M") # 0707_1723
print(date)
 
 # 3. 컴파일, 훈련
filepath = './_test/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

model.compile(loss='mae', optimizer='Adam')
start_time = time.time()
from keras.callbacks import ModelCheckpoint,EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=500, mode='min', 
                              verbose=1,restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss',mode='auto',verbose=1,
                      save_best_only=True, 
                      filepath="".join([filepath,'k46_', date, '_', filename])
                    )
model.fit([x1_train,x2_train], y1_train, 
          validation_split=0.1, 
          epochs=15,batch_size=32
          ,callbacks=[es,mcp])

end_time = time.time()

# 4. 평가, 예측
loss = model.evaluate([x1_test, x2_test], y1_test)
predict = model.predict([x1_test, x2_test])
print('loss: ', loss)
print('7/20 종가 : ', predict[-1:])
print('걸린 시간: ', end_time-start_time)

# 7/20 종가 :  [[135684.06]]
# 7/20 종가 :  [[135610.6]]
# 7/20 종가 :  [[133787.95]]
# 7/20 종가 :  [[285145.2]]