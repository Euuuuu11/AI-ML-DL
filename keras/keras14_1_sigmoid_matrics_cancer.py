import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.models import Sequential   
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = load_breast_cancer()
# print(datasets)
# print(datasets.DESCR) 
# print(datasets.feature_names)

x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(x,y,
             train_size=0.8,random_state=72)
# print(x.shape, y.shape) #(569, 30) (569, )
# print(y)

# 2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=30,activation='sigmoid')) 
model.add(Dense(80))
model.add(Dense(80))
model.add(Dense(60))
model.add(Dense(40,activation='relu'))  # 'relu' 히든 레이어에서만 사용 가능(음수는 없애고, linear와 동일)
model.add(Dense(10,activation='sigmoid'))
model.add(Dense(1,activation='sigmoid')) # 이진분류 마지막은 무조건 'sigmoid'(0 ~ 1까지)

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy']) # 2개 이상은 list

from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=500, mode='min', 
              verbose=1, restore_best_weights=True) 

hist = model.fit(x_train, y_train,
          epochs=1, batch_size=10,validation_split=0.2,
          verbose=1,  callbacks=[es] )

y_predict = model.predict(x_test)
y_predict = y_predict.round(0)
print(y_predict)

##### [과제 1.]accuracy_score 완성
from sklearn.metrics import r2_score, accuracy_score
acc = accuracy_score(y_test, y_predict)

# r2 = r2_score(y_test, y_predict)

print(y_predict)
#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc스코어 : ', acc)


# import matplotlib
# import matplotlib.pyplot as plt
# matplotlib.rcParams['font.family'] = 'Malgun Gothic'
# matplotlib.rcParams['axes.unicode_minus'] =False

# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'], marker='.',  c = 'red', label = 'loss')
# plt.plot(hist.history['val_loss'], marker='.',c = 'blue', label = 'val_loss')
# plt.grid()
# plt.title('시그모이드')
# plt.ylabel('loss')
# plt.xlabel('epochs')
# plt.legend(loc='upper right')
# plt.legend
# plt.show()

# loss :  [0.6240755915641785, 0.7719298005104065]
# acc스코어 :  0.7719298245614035


















