from tabnanny import verbose
from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np  
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets. target

x_train, x_test, y_train, y_test = train_test_split(x,y,
             train_size=0.7, shuffle=True, random_state=32)
# print(x)
# print(y)
# print(x.shape, y.shape)  # (506, 13) (506, )

print(datasets.feature_names) 
print(datasets.DESCR)


#2. 모델구성
model = Sequential()
model.add(Dense(11, input_dim=13))
model.add(Dense(22))
model.add(Dense(33))
model.add(Dense(44))
model.add(Dense(55))
model.add(Dense(66))
model.add(Dense(75))
model.add(Dense(88))
model.add(Dense(99))
model.add(Dense(1))

import time # 훈련시간을 알려준다.
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

start_time = time.time() # 시작시간
print(start_time) #1656032965.640287
model.fit(x_train, y_train, epochs=50, batch_size=1, verbose=0)

end_time = time.time() - start_time #시작시간과 끝난시간을 빼준다.
# verbose 훈련과정 안보여줌으로써 시간이 절약된다.

print("걸린시간 : ", end_time) # 빼준 결과

'''

verbose 0 걸린시간 : 12.110095262527466 / 출력없다.
verbose 1 걸린시간 : 14.63875412940979 / 잔소리 많다.
verbose 2 걸린시간 : 12.74895691871643 / 프로그래스바 없다.
verbose 3 걸린시간 : 12.65363883972168 / epoch(훈련횟수)만 나온다.
verbose 4 걸린시간 : 13.55055570602417 / epoch(훈련횟수)만 나온다.
verbose 5 걸린시간 : 12.524626731872559 / epoch(훈련횟수)만 나온다.

'''