import numpy as np  
from keras.preprocessing.image import ImageDataGenerator
from sklearn import datasets

#1. 데이터
x_train = np.load('d:/study_data/_save/_npy/keras49_5_train_x.npy')
y_train = np.load('d:/study_data/_save/_npy/keras49_5_train_y.npy')
x_test = np.load('d:/study_data/_save/_npy/keras49_5_test_x.npy')
y_test = np.load('d:/study_data/_save/_npy/keras49_5_test_y.npy')

#2. 모델 구성
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(128, (2,2), input_shape=(150,150,1), activation='relu'))
model.add(Conv2D(64, (2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# model.fit(xy_train[0][0], xy_train[0][1]) # 배치를 최대로 잡으면 이거도 가능
hist = model.fit(x_train,y_train,epochs=30,verbose=2,batch_size=32)

accuracy = hist.history['accuracy']
# val_accuracy = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss :', loss[-1])
# print('val_loss :', val_loss[-1])
print('accuracy:', accuracy[-1])
# print('val_accuracy :', val_accuracy[-1])

# 그림그려....
# import matplotlib.pyplot as plt
# plt.plot(accuracy,'gray')
# plt.show()

# loss : 3.6909264053974766e-06
# val_loss : 0.6621402502059937
# accuracy: 1.0
# val_accuracy : 0.949999988079071