import numpy as np  
from keras.preprocessing.image import ImageDataGenerator
from sklearn import datasets

#1. 데이터
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,   # 수평 반전
    vertical_flip=True,     # 수직 반전
    width_shift_range=0.1,  # 수평 이동
    height_shift_range=0.1, # 상하 이동
    rotation_range=5,       # 기울이기
    zoom_range=1.2,         # 확대
    shear_range=0.7,        # 찌그러트리기
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(  # test 데이터는 증폭할 필요가 없다.
    rescale=1./255
)
xy_data = train_datagen.flow_from_directory(   # directory = 폴더
    'D:/study_data/_data/rps/',
    target_size=(150, 150), # 크기 맞추기
    batch_size=2520,
    class_mode='categorical',
    shuffle=True,
    )
x = xy_data[0][0]
print(x)

y = xy_data[0][1]
print(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=66)

print((x_train.shape),(y_train.shape))  # (1764, 150, 150, 3) (1764, 3)
# # 현재 5,200,200,1 짜리 데이터가 32덩어리
print(arr=x_train)

np.save('d:/study_data/_save/_npy/keras47_3_train_x.npy', arr=x_train)
np.save('d:/study_data/_save/_npy/keras47_3_train_y.npy', arr=y_train)
np.save('d:/study_data/_save/_npy/keras47_3_test_x.npy', arr=x_test)
np.save('d:/study_data/_save/_npy/keras47_3_test_y.npy', arr=y_test)

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
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
hist = model.fit(x_train, y_train, epochs=30,validation_split=0.2)
# model.fit(xy_train[0][0], xy_train[0][1]) # 배치를 최대로 잡으면 이거도 가능
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

# # # 그림그려....
# # import matplotlib.pyplot as plt
# # plt.plot(accuracy,'gray')
# # plt.show()

# loss : 0.2409558743238449
# accuracy: 0.9057406187057495
