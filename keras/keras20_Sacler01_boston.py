from subprocess import call
from tabnanny import verbose
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np  
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets. target

x_train, x_test, y_train, y_test = train_test_split(x,y,
             train_size=0.8, shuffle=True, random_state=66)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

# scaler.fit(x_train)
# #print(x_train)
# x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) #x_train이작업된 범위에 맞춰서 진행
x_train = scaler.fit_transform(x_train)


#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=13, activation='relu'))
model.add(Dense(80))
model.add(Dense(80, activation='relu'))
model.add(Dense(60))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

import time
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam',metrics=['mae'])

from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=100, mode='min', 
              verbose=1, restore_best_weights=True) 

start_time = time.time()
print(start_time)

model.fit(x_train, y_train,
          epochs=1000, batch_size=1,validation_split=0.2,
          verbose=1, callbacks=[es] )

end_time = time.time() - start_time
print("걸린시간 : ", end_time)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss) 



# sklearn 에서 제공하는 스케일러 요약.
# 1	StandardScaler	기본 스케일. 평균과 표준편차 사용
# 2	MinMaxScaler	최대/최소값이 각각 1, 0이 되도록 스케일링
# 3	MaxAbsScaler	최대절대값과 0이 각각 1, 0이 되도록 스케일링
# 4	RobustScaler	중앙값(median)과 IQR(interquartile range) 사용. 아웃라이어의 영향을 최소화

# 보스턴에 대해서 3가지 비교

#1. 스케일러 하기 전
#  loss :  13.195073127746582
#  r2스코어 :  0.8421319827432074

#2. MinMaxScaler (모든 feature 값이 0~1사이에 있도록 데이터를 재조정한다. 다만 이상치가 있는경우엔 변환된 값이 매우 좁은 범위로 압축 될 수 있음. 
# MinMaxSacler역시 아웃라이어의 존재에 매우 민감.), 부동수소점 연산에 특화됨.
# loss :  7.451126575469971
# r2스코어 :  0.910853502461018

#3. StandardScaler (평균을 제거하고 데이터를 단위 분산으로 조정, 그러나 이상치가 있다면 평균과 표준편차에 영향을 미쳐 
# 변환된 데이터의 확산은 매우 달라짐. 때문에 이상치가 있는경우에는 균형잡힌 척도를 보장할 수 없다.)
# loss :  8.981207847595215
# r2스코어 :  0.8925473626465857

#4. MaxAbsScaler (절대값이 0~1 사이에 맵핑되도록 하는 것. 양수데이터로만 구성된 특징 
# 데이터셋에서는 MinMax와 유사하게 동작하며, 큰 이상치에 민감할 수 있다.)
# loss :  7.04616117477417
# r2스코어 :  0.9156985716390212

#5. RobustScaler (아웃라이어의 영향을 최소화 한 기법. 중앙값(median)과 IQR(interquartile range)를 사용하기 때문에 
# StandardScaler와 비교하면 표준화 후 동일한 값을 더 넓게 분포 시키고 있음을 확인 할 수 있음.
# * IQR = Q3 - Q1 : 25퍼센타일과 75퍼센타일의 값들을 다룸.
# loss :  8.037117958068848
# r2스코어 :  0.9038426019813244