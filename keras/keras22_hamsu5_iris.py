from tkinter.tix import Y_REGION
import numpy as np
from pandas import Categorical
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from tensorflow.python.keras.models import Sequential, Model   
from tensorflow.python.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

# import tensorflow as tf
# tf.random.set_seed(15)

#1. 데이터
datasets = load_iris()
# print(datasets.DESCR) 
# print(datasets.feature_names) # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'] 
x = datasets['data']
y = datasets.target
# print(x)
# print(y)
# print(x.shape, y.shape) # (150, 4) (150, )
print("y의 라벨값 : ", np.unique(y)) # y의 라벨값 : [0 1 2]

# ValueError: in user code:
#     ValueError: Shapes (1, 1) and (1, 3) are incompatible
# y.shape를 (150,1) -> (150,3) 으로 바꿔줘야 해결이 된다.
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

# print(y)
# print(y.shape) # (150, 3)


x_train, x_test, y_train, y_test = train_test_split(x,y,
             train_size=0.8, shuffle=True, random_state=66)


scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
#print(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# print(y_test)
# print(y_train)


#2. 모델구성
input1 = Input(shape=(4, ))
dense1 = Dense(100,activation='relu')(input1)
dense2 = Dense(80)(dense1)
dense3 = Dense(50,activation='relu')(dense2)
dense4 = Dense(50)(dense3)
dense5 = Dense(10)(dense4)
output1 = Dense(3, activation='softmax')(dense5)
model = Model(inputs = input1, outputs = output1)

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy']) # 다중분류에는 loss = 'categorical_crossentropy'만 사용 

from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=400, mode='min', 
              verbose=1, restore_best_weights=True) 

model.fit(x_train, y_train,
          epochs=500, batch_size=100,validation_split=0.2,
          verbose=1, callbacks=[es])


#4. 평가, 예측
result = model.evaluate(x_test,y_test)
print("loss : ", result[0])
print("accuracy : ", result[1])


from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
# print(y_predict)
y_test = np.argmax(y_test, axis=1)
# print(y_test)

acc = accuracy_score(y_test, y_predict)
print('acc스코어 : ', acc)



# loss :  0.03609280660748482
# acc스코어 :  1.0

# 함수형 모델 후
# loss :  0.21130409836769104
# acc스코어 :  0.9333333333333333