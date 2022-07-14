import numpy as np
from  tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, SimpleRNN, Dropout, LSTM, GRU

#1. 데이터
a = np.array(range(1, 101))
x_predict = np.array(range(96,106))

size = 5    # x는 4개, y는 1개
size2 = 4

def split_x(dataset, size,):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)
    
bbb = split_x(a, size)
ccc = split_x(x_predict, size2) # ccc라는 변수를 하나 더 만들어줘서 split_x 해준다.

# print(bbb)
# print(bbb.shape)    

x = bbb[:, :-1] #.reshape(96,4,1)
y = bbb[:, -1]
print(x,y)
print(x.shape, y.shape)
print(ccc.shape)    # (7, 4)

#2. 모델구성    
model = Sequential()
model.add(Dense(128,input_dim =(4)))    # (n,3,1) -> (n,3,170)
model.add(Dense(128,activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
# model.add(Dropout(0.2)) 
model.add(Dense(1))
model.summary() 
 
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=2002)


#4. 평가, 예측
loss = model.evaluate(x,y)
y_pred = ccc #.reshape(7, 4, 1)   
result = model.predict(y_pred)

print('loss : ', loss)
print('[96~106]의 결과 : ', result)

# LSTM
# loss :  0.14901338517665863
# [96~106]의 결과 :  [[100.71054 ]
#  [101.760574]
#  [102.81081 ]
#  [103.86116 ]
#  [104.909485]
#  [105.9311  ]
#  [106.94879 ]]

# DNN
# loss :  6.62658057990484e-05                   
# [96~106]의 결과 :  [[ 99.986404]
#  [100.98625 ]
#  [101.98613 ]
#  [102.98599 ]
#  [103.98585 ]
#  [104.98572 ]
#  [105.985565]]

# loss :  0.00025862769689410925
# [96~106]의 결과 :  [[100.02709 ]
#  [101.02736 ]
#  [102.02763 ]
#  [103.02791 ]
#  [104.028175]
#  [105.028435]
#  [106.02872 ]]