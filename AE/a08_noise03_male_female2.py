# [실습] keras47_4 남자 여자에 noise를 넣어
# predict 첫번째 : 기미 주근깨 여드름 제거
# 랜덤하게 5개정도 원본/수정본 빼고

# predict 두번째 : 본인 사진 넣어서 // 원본 수정본

import numpy as np

x_train = np.load("d:/study_data/_save/_npy/keras47_4_train_x.npy")
x_test = np.load("d:/study_data/_save/_npy/keras47_4_test_x.npy")

x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)

x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, 0, 1)

#2. model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Input, UpSampling2D
from keras.models import Sequential, Model

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(filters=hidden_layer_size, input_shape=(150, 150, 3), kernel_size=3, padding='same', strides=1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, 3, padding='same', activation='relu'))
    model.add(Conv2D(128, 3, padding='same', activation='relu'))
    model.add(UpSampling2D())
    model.add(Conv2D(3, 3, padding='same'))
    return model
    
model = autoencoder(hidden_layer_size=154)  # pca 95% -> 154

model.compile(optimizer='adam', loss='mse')

model.fit(x_train_noised, x_train, epochs=10, batch_size=16, shuffle=True, validation_split=0.2)

#3. evaluate
output = model.predict(x_test_noised)

from matplotlib import pyplot as plt
import random

fig, ((ax1, ax2, ax3, ax4, ax5), (ax11, ax12, ax13, ax14, ax15), (ax6, ax7, ax8, ax9, ax10)) = plt.subplots(3, 5, figsize=(20, 7))

# 이미지 5개를 무작위로 고른다.
random_images = random.sample(range(output.shape[0]), 5)

#원본(입력) 이미지를 맨 위에 그린다.
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]], cmap='gray')
    if i == 0:
        ax.set_ylabel("INPUT", size=40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
# 잡음을 넣은 이미지
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(x_test_noised[random_images[i]], cmap='gray')
    if i == 0:
        ax.set_ylabel("NOISE", size=40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
# 오토인코더가 출력한 이미지
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output[random_images[i]], cmap='gray')
    if i == 0:
        ax.set_ylabel("OUTPUT", size=40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()    
plt.show()


