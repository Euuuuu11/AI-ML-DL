from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np

#1. 데이터

x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델구성

model = Sequential()
model.add(Dense(16, input_dim=1))
model.add(Dense(16,activation='relu'))   
model.add(Dense(8,activation='relu'))    
model.add(Dense(4,activation='relu'))    
model.add(Dense(1))    
 
model.summary() #연산량을 보여준다
#y=wx+b 에서 바이어스(b)가 1개의 추가 노드를 차지


# Model: "sequential"
# _________________________________________________________________     
# Layer (type)                 Output Shape              Param #        
# =================================================================     
# dense (Dense)                (None, 5)                 10
# _________________________________________________________________     
# dense_1 (Dense)              (None, 3)                 18
# _________________________________________________________________     
# dense_2 (Dense)              (None, 4)                 16
# _________________________________________________________________     
# dense_3 (Dense)              (None, 2)                 10
# _________________________________________________________________     
# dense_4 (Dense)              (None, 1)                 3
# =================================================================     
# Total params: 57
# Trainable params: 57
# Non-trainable params: 0
# _________________________________________________________________