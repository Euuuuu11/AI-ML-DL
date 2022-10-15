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
xy_train = train_datagen.flow_from_directory(   # directory = 폴더
    'd:/study_data/_data/image/brain/trian/',
    target_size=(100, 100), # 크기 맞추기
    batch_size=160,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True,
    )   # Found 160 images belonging to 2 classes.
   
xy_test = test_datagen.flow_from_directory(   # directory = 폴더
    'd:/study_data/_data/image/brain/test/',
    target_size=(100, 100), # 크기 맞추기
    batch_size=120,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True,
    )   # Found 120 images belonging to 2 classes.

# print(xy_train) # <keras.preprocessing.image.DirectoryIterator object at 0x000001E9E21D2F40>

# from sklearn.datasets import load_boston
# datasets = load_boston()
# print(datasets)

# print(xy_train[31])  # x,y 값 둘다 포함되어있다.
# ValueError: Asked to retrieve element 33, but the Sequence has length 32
# = 총 160개의 데이터가 있고 배치사이즈 5개 단위로 잘렸을 때 32개의 데이터가 있는데 33개 데이터 요청, # 0 ~ 31까지 가능.
print(xy_train[0][0].shape)    # (5, 200, 200, 1)
print(xy_train[0][1])          # [0. 1. 0. 1. 1.] = y값    
# print(xy_train[31][2])    # IndexError: tuple index out of range)

print((xy_train[0][0].shape),(xy_train[0][1].shape))

print(type(xy_train))   # <class 'keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0]))    # <class 'tuple'>
print(type(xy_train[0][0]))    # <class 'numpy.ndarray'>
print(type(xy_train[0][1]))    # <class 'numpy.ndarray'>

# 현재 5,200,200,1 짜리 데이터가 32덩어리

#2. 모델 구성
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(128, (2,2), input_shape=(100,100,1), activation='relu'))
model.add(Conv2D(64, (2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

hist = model.fit(xy_train[0][0], xy_train[0][1], epochs=30,validation_split=0.2)  # 배치를 최대로 잡으면 이거도 가능
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

# 그림그려....
import matplotlib.pyplot as plt
plt.plot(accuracy,'gray')
plt.show()

# loss : 0.00362432561814785
# val_loss : 0.9042067527770996
# accuracy: 1.0
# val_accuracy : 0.65625