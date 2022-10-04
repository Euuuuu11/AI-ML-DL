import numpy as np  
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

#1.데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
            [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4],
            [9,8,7,6,5,4,3,2,1,0]]             
            )
y = np.array([11,12,13,14,15,16,17,18,19,20])

print(x.shape,y.shape) # (3, 10)  (10, ) 
x = x.T
print(x.shape)    # (10, 3)

#2. 모델구성
model = Sequential()
# model.add(Dense(10,input_dim=3))   # (100, 3) -> (None, 3)
model.add(Dense(16,input_shape=(3,)))  
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))
model.summary()

# Model: "sequential"     # None = 행의 갯수
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense (Dense)                (None, 16)                64
# _________________________________________________________________
# dense_1 (Dense)              (None, 8)                 136
# _________________________________________________________________
# dense_2 (Dense)              (None, 4)                 36
# _________________________________________________________________
# dense_3 (Dense)              (None, 1)                 5
# =================================================================
# Total params: 241
# Trainable params: 241
# Non-trainable params: 0
# _________________________________________________________________