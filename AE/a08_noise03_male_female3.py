# [실습] keras 47_4 남자 여자에 noise 넣고 
# 기미 주끈개 여드름 제거!!

# predict 두번쨰 : 본인 사진넣기

import aifc
import numpy as np      
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn import datasets

x_train = np.load("d:/study_data/_save/_npy/keras47_4_train_x.npy")
y_train = np.load("d:/study_data/_save/_npy/keras47_4_train_y.npy")
x_test = np.load("d:/study_data/_save/_npy/keras47_4_test_x.npy")
y_test = np.load("d:/study_data/_save/_npy/keras47_4_test_y.npy")
z_test = np.load('D:/study_data/_save/_npy/keras100_4_test_set.npy')
# print(x_train.shape, x_test.shape) # (350, 150, 150, 3) (150, 150, 150, 3)

# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
x_train_noised = x_train + np.random.normal(0, 0.1 , size = x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1 , size=x_test.shape)
z_test_noised = z_test + np.random.normal(0, 0.1 , size=x_test.shape)


x_train_noised = np.clip(x_train_noised, a_min=0,a_max=1) # noised는 최댓값이 1.1이기 때문에 범위 조정
x_test_noised = np.clip(x_test_noised, a_min=0,a_max=1)
z_test_noised = np.clip(z_test_noised, a_min=0,a_max=1)

#2. 모델 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D,Flatten,Dense,MaxPool2D,UpSampling2D

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(hidden_layer_size, (3, 3), activation='relu', padding='same', strides=2,input_shape=(150,150,3)))
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same'))
    # model.compile(optimizer='rmsprop', loss='mse')
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

model = autoencoder(hidden_layer_size=320)

model.fit(x_train_noised, x_train, epochs=50, batch_size=128,
                validation_split=0.2)
output = model.predict(x_test)
output2 = model.predict(z_test)
# print(output.shape)


from matplotlib import pyplot as plt
import random 

fig, ((ax1, ax2, ax3, ax4,ax5, image1), (ax6, ax7, ax8, ax9, ax10,image2),
      (ax11, ax12, ax13, ax14, ax15,image3))  = \
    plt.subplots(3, 6, figsize=(20, 7))
    
# 이미지 다섯 개를 무작위로 고른다.
random_images = random.sample(range(output.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(150,150,3))
    if i ==0:
        ax.set_ylabel("INPUT", size=20)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
image1.imshow(z_test[0])   
     
# noise를 넣은 이미지
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test_noised[random_images[i]].reshape(150,150,3))
    if i ==0:
        ax.set_ylabel("NOISE", size=20)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])        
image2.imshow(z_test_noised[0])   

        
# 오토인코더가 출력한 이미지를 아래에 그린다
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(output[random_images[i]].reshape(150,150,3))
    if i ==0:
        ax.set_ylabel("OUTPUT", size=20)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
image3.imshow(output2[0])   
        
plt.tight_layout()
plt.show()    

