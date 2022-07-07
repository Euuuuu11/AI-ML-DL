import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.models import Sequential, Model   
from tensorflow.python.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

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

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)
#print(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델구성
input1 = Input(shape=(30,))   
dense1 = Dense(100, input_dim=30,activation='sigmoid')(input1)
dense2 = Dense(80)(dense1)
dense3 = Dense(80)(dense2)
dense4 = Dense(60)(dense3)
dense5 = Dense(40,activation='relu')(dense4) # 'relu' 히든 레이어에서만 사용 가능(음수는 없애고, linear와 동일)
dense6 = Dense(10,activation='sigmoid')(dense5)
output1 = Dense(1,activation="sigmoid")(dense6) # 이진분류 마지막은 무조건 'sigmoid'(0 ~ 1까지), loss='binary_crossentropy'
model = Model(inputs=input1, outputs=output1)

model.save("./_save/kersa23_10_save_model_cancer.h5")

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy']) # 2개 이상은 list

from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=500, mode='min', 
              verbose=1, restore_best_weights=True) 

hist = model.fit(x_train, y_train,
          epochs=100, batch_size=10,validation_split=0.2,
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


#5. RobustScaler 
#  loss :  0.16829858720302582
#  acc스코어 :   0.9649122807017544

# 함수형 모델 후
# loss :  0.1570216417312622
# acc스코어 :  0.9649122807017544
