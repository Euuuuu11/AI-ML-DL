# 넘파이에서 불러와서 모델 구성
# 성능비교
from tensorflow.keras.datasets import mnist,fashion_mnist, cifar10, cifar100
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Conv1D, Flatten, MaxPooling2D, Conv2D   



#1. 데이터
x_train = np.load('d:/study_data/_save/_npy/keras49_4_train_x.npy')
y_train = np.load('d:/study_data/_save/_npy/keras49_4_train_y.npy')
x_test = np.load('d:/study_data/_save/_npy/keras49_4_test_x.npy')
y_test = np.load('d:/study_data/_save/_npy/keras49_4_test_y.npy')


#2. 모델 구성
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D,Flatten,Dense,MaxPool2D

model = Sequential()
model.add(Conv1D(8, 2, input_shape=(32,32,3)))
model.add(Flatten())
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(100, activation="softmax"))

#3. 컴파일,훈련
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

hist = model.fit(x_train,y_train,epochs=30,verbose=2,validation_split=0.25,batch_size=128)
# hist = model.fit_generator(x_train,y_train,epochs=2,
#                     validation_split=0.25,
#                     steps_per_epoch=32,
#                     validation_steps=4) # 배치가 최대 아닐 경우 사용


#4. 평가,예측
accuracy = hist.history['accuracy']
val_accuracy = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss :', loss[-1])
print('val_loss :', val_loss[-1])
print('accuracy:', accuracy[-1])
print('val_accuracy :', val_accuracy[-1])

# loss : 3.446493148803711
# accuracy: 0.19868148863315582