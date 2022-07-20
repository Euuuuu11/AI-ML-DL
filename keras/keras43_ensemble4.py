# Ensemble
#1. 데이터
import numpy as np
from sklearn.decomposition import KernelPCA
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Input
x1_datasets = np.array([range(100), range(301,401)])    # 삼전 종가, 하이닉스 종가

x1 = np.transpose(x1_datasets)


# print(x1_datasets.shape, x2_datasets.shape)   # (100, 2) (100, 3)
y1 = np.array(range(2001, 2101)) # 금리
y2 = np.array(range(201, 301)) # 금리 
 

from sklearn.model_selection import train_test_split
x1_train, x1_test,y1_train, y1_test, y2_train, y2_test = train_test_split(
    x1, y1, y2, train_size=0.7, random_state=66)

# print(x1_train.shape, x1_test.shape)    # (70, 2) (30, 2)
# print(x2_train.shape, x2_test.shape)    # (70, 3) (30, 3)
# print(x3_train.shape, x3_test.shape)    # (70, 2) (30, 2)
# print(y1_train.shape, y1_test.shape)    # (70, ) (30, )
# print(y2_train.shape, y2_test.shape)    # (70, ) (30, )

#2. 모델구성
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Input

#2-1. 모델1
input1 = Input(shape=(2,))
dense1 = Dense(256, activation='relu', name = 'jg1') (input1)
dense2 = Dense(128, activation='relu', name = 'jg2') (dense1)
dense3 = Dense(64, activation='relu', name = 'jg3') (dense2)
output1 = Dense(32, activation='relu', name = 'out_jg1') (dense3)

# #2-2. 모델2
# input2 = Input(shape=(3,))
# dense11 = Dense(256, activation='relu', name = 'jg11') (input2)
# dense12 = Dense(128, activation='relu', name = 'jg12') (dense11)
# dense13 = Dense(64, activation='relu', name = 'jg13') (dense12)
# dense14 = Dense(32, activation='relu', name = 'jg14') (dense13)
# output2 = Dense(16, activation='relu', name = 'out_jg2') (dense14)


# #2-3. 모델3
# input3 = Input(shape=(2,))                                      
# dense111 = Dense(256, activation='relu', name = 'jg111') (input3)
# dense112 = Dense(128, activation='relu', name = 'jg112') (dense111)
# dense113 = Dense(64, activation='relu', name = 'jg113') (dense112)
# dense114 = Dense(32, activation='relu', name = 'jg114') (dense113)
# output3 = Dense(16, activation='relu', name = 'out_jg3') (dense114)

# Concatenate
from tensorflow.python.keras.layers import concatenate, Concatenate # 연산하지 않고, 연결만 해준다.
# merge1 = concatenate([output1, output2,output3], name='m1')
# merge1 = Concatenate()([output1, output2,output3])
# merge2 = Dense(16, activation='relu', name='m2')(merge1)
# merge3 = Dense(8, name='m3')(merge2)
# last_output = Dense(1, name='last') (merge3)

#2-4. output모델1
output41 = Dense(10)(output1)
output42 = Dense(10)(output41)
last_output2 = Dense(1)(output42)

#2-5. output모델1
output51 = Dense(10)(output1)
output52 = Dense(10)(output51)
output53 = Dense(10)(output52)
last_output3 = Dense(1)(output53)

model = Model(inputs=[input1], outputs=[last_output2, last_output3])

# merge11 = concatenate([output1, output2,output3], name='m11')
# merge12 = Dense(16, activation='relu', name='m12')(merge11)
# merge13 = Dense(8, name='m13')(merge12)
# last_output1 = Dense(1, name='last2') (merge13)
# model.summary()
# print(y1.shape,y2.shape) (100,) (100,)

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')   
                                      
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=60, mode='auto', verbose=1, 
                              restore_best_weights=True)        

model.fit([x1_train], [y1_train,y2_train], epochs=1005, batch_size=32,     # 두개 이상은 list이므로, list형식으로 해준다.
                 validation_split=0.2,
                 callbacks=[es],
                 verbose=1)

#4. 평가,예측

loss = model.evaluate([x1_test],[y1_test,y1_test])
result1, result2 = model.predict([x1_test]) 

from sklearn.metrics import r2_score
r2_2 = r2_score(y1_test,result1)    
r2_1 = r2_score(y2_test,result2) 
print('loss : ', loss)
print('r2_1스코어 : ', r2_1 )
print('r2_2스코어 : ', r2_2 )

# loss :  [3239996.25, 5.674958083545789e-05, 3239996.25]
# r2_1스코어 :  0.9999999735706195
# r2_2스코어 :  0.9999999350975113

# ======================================================================================
# loss = model.evaluate([x1_test,x2_test,x3_test],[y1_test,y2_test])
# result = model.predict([x1_test,x2_test,x3_test])

# result = np.array(result)
# y_test = np.array([y1_test, y2_test])
# # print(result.shape) # (2, 30, 1)
# result = result.reshape(2, 30)
# y_test = y_test.reshape(2, 30)
# from sklearn.metrics import r2_score
# r2 = r2_score(y_test,result)

# print('loss : ', loss)
# print('r2스코어 : ', r2 )

# loss :  [0.007140777073800564, 0.0033853172790259123, 0.0037554597947746515]
# r2스코어 :  0.999999995592113






# loss_1 :  [3239922.75, 0.0011286536464467645, 3239922.75]
# r2_1스코어 :  0.9999892981834955
# loss_2 :  [3240001.25, 3240001.25, 0.009357478469610214]
# r2_2스코어 :  0.9999987091987274
