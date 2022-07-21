from re import X
from tensorflow.keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    # vertical_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=5,
    zoom_range=0.1,
    # shear_range=0.7,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(
    rescale=1./255,)

augument_size = 10 # 증폭                                            # https://codetorial.net/numpy/random.html
randidx = np.random.randint(x_train.shape[0], size=augument_size)       # 함수는 [최소값, 최대값)의 범위에서 임의의 정수를 만듬.
# print(x_train.shape[0]) # 60000
# print(randidx)  # [22736 14506 25834 ... 57205 57634  3909]
# print(np.min(randidx), np.max(randidx)) # 1 59997
# print(type(randidx))    # <class 'numpy.ndarray'>
x_train1 = x_train[randidx].copy()
x_augumented = x_train[randidx].copy()  # .copy() 
y_augumented = y_train[randidx].copy() 
# print(x_augumented.shape)   # (40000, 28, 28)
# print(y_augumented.shape)   # (40000,)
x_train1 = x_train1.reshape(10, 28, 28, 1)

# x_test = x_test.reshape(x_test.shape[0],x_test.shape[1], x_test.shape[2],1)
x_augumented = x_augumented.reshape(x_augumented.shape[0],
                                    x_augumented.shape[1], 
                                    x_augumented.shape[2],1)




# 변환
x_augumented = train_datagen.flow(x_augumented, y_augumented,
                                  batch_size=augument_size,
                                  shuffle=False).next()[0]  # shuffle=False 하는 이유 randidx로 랜덤으로 가져왔기 때문.

# print(x_augumented)
# print(x_augumented.shape)   (40000, 28, 28, 1)
x_train1 = np.concatenate((x_train1, x_augumented))
# x_train = np.concatenate((x_train, x_augumented))
# y_train = np.concatenate((y_train, y_augumented))
# print(x_train.shape, y_train.shape) # (100000, 28, 28, 1) (100000,)

print(x_train[0].shape) # (28, 28)
print(x_train[0].reshape(28*28).shape) # (784,)                                         # np.tile =  배열 쌓기 함수
print(np.tile(x_train1[0].reshape(28*28), augument_size).reshape(-1, 28, 28, 1).shape)   # (40000, 28, 28, 1)
# print(np.zeros(augument_size))
# print(np.zeros(augument_size).shape)    # (100,)

x_data = test_datagen.flow(
    x_train1.reshape(-1, 28, 28, 1),    # x
    np.zeros(20),                                                     # y
    batch_size=20,
    shuffle=False,
).next()    # https://offbyone.tistory.com/83

#################################### .next() 사용 ##########################################
print(x_data)   
print(x_data[0])            
print(x_data[0].shape)   
print(x_data[1].shape)

#################################### .next() 미사용 ##########################################
# print(x_data)   # <keras.preprocessing.image.NumpyArrayIterator object at 0x000002011F21CD90>
# print(x_data[0])            # x,y 모두 포함
# print(x_data[0][0].shape)   # (100, 28, 28, 1)
# print(x_data[0][1].shape)   # (100,)

import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
for i in range(20):
    plt.subplot(2, 10, i+1)
    plt.axis('off')
    # plt.imshow(x_data[0][i], cmap='gray')    # .next 사용
    plt.imshow(x_data[0][i], cmap='gray')    # .next 미사용
plt.show()    

# [실습]
# x_augumented 10개와 x_train 10개를 비교하는 이미지 출력할 것!!!




