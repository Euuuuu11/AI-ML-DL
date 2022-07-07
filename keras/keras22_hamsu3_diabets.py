from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
import numpy as np  
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

datasets = load_diabetes()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(x,y,
             train_size=0.8,random_state=72)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)
#print(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# print(x)
# print(y)
# print(x.shape, y.shape) # (442, 10) (442, )

# print(datasets.feature_names)    # ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
# print(datasets.DESCR)   

#2. 모델구성
input1 = Input(shape=(10,))
dense1 = Dense(10)(input1)
dense2 = Dense(50,activation='relu')(dense1)
dense3 = Dense(40,activation='relu')(dense2)
dense4 = Dense(50,activation='relu')(dense3)
dense5 = Dense(60,activation='relu')(dense4)
dense6 = Dense(20,activation='relu')(dense5)
dense7 = Dense(10,activation='relu')(dense6)
dense8 = Dense(10,activation='relu')(dense7)
output1 = (Dense(1))(dense8)
model = Model(inputs=input1, outputs=output1)

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam',metrics=['mae'])

from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=400, mode='min', 
              verbose=1, restore_best_weights=True) 
model.fit(x_train, y_train, epochs=1000, validation_split=0.2,
                 batch_size=15, verbose=1, callbacks=[es])

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)






#  RobustScaler 
#  loss :  2578.9599609375
#  r2스코어 :   0.6094389508664131

# 함수형 모델 후
# loss :  2306.437255859375
# r2스코어 :  0.6507101320579136

