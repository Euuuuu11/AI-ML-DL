# Ensemble
#1. 데이터
import numpy as np
from sklearn.decomposition import KernelPCA
x1_datasets = np.array([range(100), range(301,401)])    # 삼전 종가, 하이닉스 종가
x2_datasets = np.array([range(101, 201), range(411,511), range(150,250)])   # 원유, 돈육, 밀
x3_datasets = np.array([range(100, 200), range(1301,1401)])    # 우리반 아이큐, 우리반 키
x1 = np.transpose(x1_datasets)
x2 = np.transpose(x2_datasets)
x3 = np.transpose(x3_datasets)

# print(x1_datasets.shape, x2_datasets.shape)   # (100, 2) (100, 3)
y = np.array(range(2001, 2101)) # 금리 

from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test,x3_train, x3_test, y_train, y_test = train_test_split(
    x1, x2,x3, y, train_size=0.7, random_state=66)

# print(x1_train.shape, x1_test.shape)    # (70, 2) (30, 2)
# print(x2_train.shape, x2_test.shape)    # (70, 3) (30, 3)
# print(y_train.shape, y_test.shape)      # (70, ) (30, )

#2. 모델구성
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Input

#2-1. 모델1
input1 = Input(shape=(2,))
dense1 = Dense(256, activation='relu', name = 'jg1') (input1)
dense2 = Dense(128, activation='relu', name = 'jg2') (dense1)
dense3 = Dense(64, activation='relu', name = 'jg3') (dense2)
output1 = Dense(32, activation='relu', name = 'out_jg1') (dense3)

#2-2. 모델2
input2 = Input(shape=(3,))
dense11 = Dense(256, activation='relu', name = 'jg11') (input2)
dense12 = Dense(128, activation='relu', name = 'jg12') (dense11)
dense13 = Dense(64, activation='relu', name = 'jg13') (dense12)
dense14 = Dense(32, activation='relu', name = 'jg14') (dense13)
output2 = Dense(16, activation='relu', name = 'out_jg2') (dense14)

#2-3. 모델3
input3 = Input(shape=(2,))
dense111 = Dense(256, activation='relu', name = 'jg111') (input3)
dense112 = Dense(128, activation='relu', name = 'jg112') (dense111)
dense113 = Dense(64, activation='relu', name = 'jg113') (dense112)
dense114 = Dense(32, activation='relu', name = 'jg114') (dense113)
output3 = Dense(16, activation='relu', name = 'out_jg3') (dense114)

from tensorflow.python.keras.layers import concatenate, Concatenate # 연산하지 않고, 연결만 해준다.
merge1 = concatenate([output1, output2,output3], name='m1')
merge2 = Dense(16, activation='relu', name='m2')(merge1)
merge3 = Dense(8, name='m3')(merge2)
last_output = Dense(1, name='last') (merge3)
model = Model(inputs=[input1, input2,input3], outputs=last_output)
# model.summary()


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')   
                                      
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=60, mode='auto', verbose=1, 
                              restore_best_weights=True)        

model.fit([x1_train, x2_train,x3_train], y_train, epochs=1002, batch_size=32,     # 두개 이상은 list이므로, list형식으로 해준다.
                 validation_split=0.2,
                 callbacks=[es],
                 verbose=1)

# #4. 평가,예측
loss = model.evaluate([x1_test,x2_test,x3_test],y_test)
result = model.predict([x1_test,x2_test,x3_test]) 

from sklearn.metrics import r2_score
r2 = r2_score(y_test, result)

print('loss : ', loss)
print('r2스코어 : ', r2 )

# loss :  0.015904607251286507
# r2스코어 :  0.9999818104623303


















