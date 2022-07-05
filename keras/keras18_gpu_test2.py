import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.models import Sequential   
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if(gpus) : 
    print("쥐피유 돈다")
    aaa = 'gpu'
else:
    print("쥐피유 안도라")
    aaa = 'cpu'
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
model = Sequential()     # activation? 활성화 함수
model.add(Dense(500, input_dim=30,activation='relu')) 
model.add(Dense(300,activation='relu'))
model.add(Dense(300))
model.add(Dense(400,activation='relu'))
model.add(Dense(300,activation='relu'))
model.add(Dense(400))
model.add(Dense(1,activation='sigmoid')) # 이진분류 마지막은 무조건 'sigmoid'(0 ~ 1까지), loss='binary_crossentropy'

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy']) # 2개 이상은 list
import time
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=200, mode='min', 
              verbose=1, restore_best_weights=True) 
start_time = time.time()
hist = model.fit(x_train, y_train,
          epochs=100, batch_size=1,validation_split=0.2,
          verbose=1,  callbacks=[es] )
end_time = time.time()-start_time
y_predict = model.predict(x_test)
y_predict = y_predict.round(0)
print(y_predict)

##### [과제 1.]accuracy_score 완성
from sklearn.metrics import r2_score, accuracy_score
acc = accuracy_score(y_test, y_predict)

# r2 = r2_score(y_test, y_predict)

# print(y_predict)
#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc스코어 : ', acc)

print(aaa, "걸린시간 : ", end_time)



# CPU 걸린시간 :  59.02738904953003
# GPU 걸린시간 :  157.01645827293396
# 작은 데이터는 CPU가 빠르다.











