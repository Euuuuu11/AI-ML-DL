from tensorflow.keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    shear_range=0.7,
    fill_mode='nearest'
)

augument_size = 100 # 증폭

# print(x_train[0].shape) # (28, 28)
# print(x_train[0].reshape(28*28).shape) # (784,)                                         # np.tile =  배열 쌓기 함수
# print(np.tile(x_train[0].reshape(28*28), augument_size).reshape(-1, 28, 28, 1).shape)   # (100, 28, 28, 1)
# print(np.zeros(augument_size))
# print(np.zeros(augument_size).shape)    # (100,)

x_data = train_datagen.flow(
    np.tile(x_train[0].reshape(28*28), augument_size).reshape(-1, 28, 28, 1),    # x
    np.zeros(augument_size),                                                     # y
    batch_size=augument_size,
    shuffle=True,
)#.next()    # https://offbyone.tistory.com/83

#################################### .next() 사용 ##########################################
# print(x_data)   
# print(x_data[0])            
# print(x_data[0].shape)   
# print(x_data[1].shape)

#################################### .next() 미사용 ##########################################
print(x_data)   # <keras.preprocessing.image.NumpyArrayIterator object at 0x000002011F21CD90>
print(x_data[0])            # x,y 모두 포함
print(x_data[0][0].shape)   # (100, 28, 28, 1)
print(x_data[0][1].shape)   # (100,)

import matplotlib.pyplot as plt
plt.figure(figsize=(7,7))
for i in range(49):
    plt.subplot(7, 7, i+1)
    plt.axis('off')
    # plt.imshow(x_data[0][i], cmap='gray')    # .next 사용
    plt.imshow(x_data[0][0][i], cmap='gray')    # .next 미사용
plt.show()    














