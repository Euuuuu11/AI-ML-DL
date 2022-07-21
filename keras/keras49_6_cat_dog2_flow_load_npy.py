import numpy as np  
from keras.preprocessing.image import ImageDataGenerator
from sklearn import datasets

#1. 데이터
x_train = np.load('d:/study_data/_save/_npy/keras49_6_train_x.npy')
y_train = np.load('d:/study_data/_save/_npy/keras49_6_train_y.npy')
x_test = np.load('d:/study_data/_save/_npy/keras49_6_test_x.npy')
y_test = np.load('d:/study_data/_save/_npy/keras49_6_test_y.npy')
# print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
# (160, 150, 150, 1) (160,) (120, 150, 150, 1) (120,)

# 현재 5,200,200,1 짜리 데이터가 32덩어리

#2. 모델 구성
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(128, (2,2), input_shape=(150,150,3), activation='relu'))
model.add(Conv2D(64, (2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(2, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

hist = model.fit(x_train, y_train, epochs=30,validation_split=0.2)
# hist = model.fit_generator(xy_train, epochs=30, steps_per_epoch=32,  # 한 epoch 종료 시 마다 검증할 때 사용되는 검증 스텝 수를 지정   
#                                           # 전체데이터/batch = 160/5 = 32
#                     validation_data=xy_test,
#                     validation_steps=4)
accuracy = hist.history['accuracy']
val_accuracy = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss :', loss[-1])
print('val_loss :', val_loss[-1])
print('accuracy:', accuracy[-1])
print('val_accuracy :', val_accuracy[-1])

# # 그림그려....
# import matplotlib.pyplot as plt
# plt.plot(accuracy,'gray')
# plt.show()

# loss : 0.042279619723558426
# accuracy: 0.9825000166893005

# loss : 0.03363071382045746
# accuracy: 0.9907407164573669