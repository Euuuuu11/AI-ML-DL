import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential   
from tensorflow.python.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target
print(x.shape,y.shape) # (178, 13) (178, )
print(np.unique(y,return_counts=True))    # [0 1 2]
# (array([0, 1, 2]), array([59, 71, 48], dtype=int64))

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y.shape) # (178, 3)

x_train, x_test, y_train, y_test = train_test_split(x,y,
             train_size=0.8, shuffle=True, random_state=66)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)
print(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=13,activation='relu'))
model.add(Dense(80))
model.add(Dense(60,activation='relu'))
model.add(Dense(60))
model.add(Dense(10,activation='relu'))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=400, mode='min', 
              verbose=1, restore_best_weights=True) 

model.fit(x_train, y_train,
          epochs=500, batch_size=100,validation_split=0.2,
          verbose=1, callbacks=[es])

#4. 평가,예측
result = model.evaluate(x_test,y_test)
print("loss : ", result[0])
print("accuracy : ", result[1])

print("============================================") 

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test, axis=1)

acc = accuracy_score(y_test, y_predict)
print('acc스코어 : ', acc)

#1. 스케일러 하기 전
#  loss :  1.0959184169769287
#  acc스코어 : 0.3888888888888889

#2. MinMaxScaler 
# loss :   0.013889981433749199
# acc스코어 : 1.0

#3. StandardScaler  
# loss :  0.03188563510775566
# acc스코어 :    0.9722222222222222

#4. MaxAbsScaler 
# loss :  0.001527016400359571
# acc스코어 :   1.0

#5. RobustScaler 
#  loss :  0.06010664254426956
#  acc스코어 : 0.9722222222222222