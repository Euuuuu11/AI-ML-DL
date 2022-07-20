import pandas as pd
from tensorflow.python.keras.models import Sequential, Model, load_model
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

# print(df.shape, df1.shape)  
# Dtype 변형
df = df.astype(dtype='float64')
df1 = df1.astype(dtype='float64')

feature_cols = ['high', 'low', 'close', 'volume','diff','wk Range','month','day']
label_cols = ['open']
  
SIZE = 5
def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)


x1 = split_x(df1[feature_cols], SIZE)
y = split_x(df1[label_cols], SIZE)
x2 = split_x(df[feature_cols], SIZE)

# print(x1.shape, y.shape, x2.shape)  # (3166, 5, 8) (3166, 5, 1) (3166, 5, 8)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scale_cols = ['open', 'high', 'low', 'close', 'volume','diff','wk Range','month','day']
df_scaled = scaler.fit_transform(df[scale_cols])

df_scaled = pd.DataFrame(df_scaled)
df_scaled.columns = scale_cols
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, test_size=0.2, shuffle=False)

# print(x1_train.shape, x1_test.shape)    # (2532, 5, 8) (634, 5, 8)
# print(x2_train.shape, x2_test.shape)    # (2532, 5, 8) (634, 5, 8)
# print(y_train.shape, y_test.shape)      # (2532, 5, 1) (634, 5, 1)

# x1_train = x1_train.reshape(2532*5,8)
# x1_train = scaler.fit_transform(x1_train)
# x1_test = x1_test.reshape(634*5,8)
# x1_test = scaler.transform(x1_test)

# x2_train = x2_train.reshape(2532*5,8)
# x2_train = scaler.fit_transform(x2_train)
# x2_test = x2_test.reshape(634*5,8)
# x2_test = scaler.transform(x2_test)

# y_train = y_train.reshape(2532*5,1)
# y_train = scaler.fit_transform(y_train)
# y_test = y_test.reshape(634*5,1)
# y_test = scaler.transform(y_test)

# print(x1_train.shape, x1_test.shape)    # (12660, 8) (3170, 8)
# print(x2_train.shape, x2_test.shape)    # (12660, 8) (3170, 8)   
# print(y_train.shape, y_test.shape)      # (12660, 1) (3170, 1) 

# # 2. 모델구성
# # 2-1. 모델1
# input1 = Input(shape=(5, 8))
# dense1 = Conv1D(64, 2, activation='relu', name='d1')(input1)
# dense2 = LSTM(128, activation='relu', name='d2')(dense1)
# dense3 = Dense(64, activation='relu', name='d3')(dense2)
# output1 = Dense(32, activation='relu', name='out_d1')(dense3)

# # 2-2. 모델2
# input2 = Input(shape=(5, 8))
# dense11 = Conv1D(64, 2, activation='relu', name='d11')(input2)
# dense12 = LSTM(128, activation='swish', name='d12')(dense11)
# dense13 = Dense(64, activation='relu', name='d13')(dense12)
# dense14 = Dense(32, activation='relu', name='d14')(dense13)
# output2 = Dense(16, activation='relu', name='out_d2')(dense14)

# from tensorflow.python.keras.layers import concatenate
# merge1 = concatenate([output1, output2], name='m1')
# merge2 = Dense(100, activation='relu', name='mg2')(merge1)
# merge3 = Dense(100, name='mg3')(merge2)
# last_output = Dense(1, name='last')(merge3)

# model = Model(inputs=[input1, input2], outputs=[last_output])
# import datetime
# date = datetime.datetime.now()
# print(date)

# date = date.strftime("%m%d_%H%M") # 0707_1723
# print(date)
 
#  # 3. 컴파일, 훈련
# filepath = './_ModelCheckPoint/K46_1/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

# model.compile(loss='mse', optimizer='Adam')
# start_time = time.time()
# from keras.callbacks import ModelCheckpoint,EarlyStopping
# es = EarlyStopping(monitor='val_loss', patience=500, mode='min', 
#                               verbose=1,restore_best_weights=True)
# mcp = ModelCheckpoint(monitor='val_loss',mode='auto',verbose=1,
#                       save_best_only=True, 
#                       filepath="".join([filepath,'k46_', date, '_', filename])
#                     )
# model.fit([x1_train,x2_train], y_train, 
#           validation_split=0.1, 
#           epochs=30,batch_size=64
#           ,callbacks=[es,mcp])

# end_time = time.time()
model = load_model('./_ModelCheckpoint/K46_1/k46_0718_1919_0030-36112536.0000.hdf5')
# 4. 평가, 예측
loss = model.evaluate([x1_test, x2_test], y_test)
predict = model.predict([x1_test, x2_test])
print('loss: ', loss)
print('7/19 시가 : ', predict[-1:])
# print('걸린 시간: ', end_time-start_time)

# 7/19 시가 :  [[134905.81]]
