from subprocess import call
from tabnanny import verbose
from sklearn.datasets import load_boston
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np  
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets. target

x_train, x_test, y_train, y_test = train_test_split(x,y,
             train_size=0.8, shuffle=True, random_state=66)
# print(x)
# print(y)
# print(x.shape, y.shape)  # (506, 13) (506, )

# print(datasets.feature_names) 
# print(datasets.DESCR)


#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=13, activation='selu'))
model.add(Dense(80))
model.add(Dense(80, activation='selu'))
model.add(Dense(60))
model.add(Dense(10, activation='selu'))
model.add(Dense(1))

import time
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=10, mode='min', 
              verbose=1, restore_best_weights=True) 

start_time = time.time()
print(start_time)

hist = model.fit(x_train, y_train,
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

# print("========================")
# print(hist) 
# # <tensorflow.python.keras.callbacks.History object at 0x00000167B1367D60>
# print("========================")
# print(hist.history) # 키 밸류 형태로 'loss'와 'val_loss를 반환해준다.
# {'loss': [1927.3525390625, 108.69204711914062, 99.18246459960938, 98.81649017333984, 96.68382263183594, 90.43391418457031, 94.44075012207031, 91.16068267822266, 90.75550079345703, 85.99461364746094, 75.10645294189453], 
#'val_loss': [86.81536102294922, 122.40486145019531, 109.92891693115234, 79.39165496826172, 93.984130859375, 76.79365539550781, 80.1338119506836, 66.88794708251953, 71.1214828491211, 85.90555572509766, 123.93794250488281]}
# print("========================")
# print(hist.history['loss'])
# print("========================")
# print(hist.history['val_loss'])

import matplotlib.pyplot as plt
import matplotlib  # 한글 깨짐 방지
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] =False

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], marker='.',  c = 'red', label = 'loss')
plt.plot(hist.history['val_loss'], marker='.',c = 'blue', label = 'val_loss')
plt.grid()
plt.title('보스턴')
plt.ylabel('loss')
plt.xlabel('epochs')
# plt.legend(loc='upper right') # 라벨 값 명칭의 위치
plt.legend
plt.show()


# loss :  17.505720138549805
# r2스코어 :  0.78811021977448

# EarlyStopping 후
# loss :  13.131475448608398
# r2스코어 :  0.8428928454653812