from tensorflow.keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    # vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    # shear_range=0.7,
    fill_mode='nearest'
)

augument_size = 40000 # 증폭                                            # https://codetorial.net/numpy/random.html
randidx = np.random.randint(x_train.shape[0], size=augument_size)       # 함수는 [최소값, 최대값)의 범위에서 임의의 정수를 만듬.
# print(x_train.shape[0]) # 60000
# print(randidx)  # [22736 14506 25834 ... 57205 57634  3909]
# print(np.min(randidx), np.max(randidx)) # 1 59997
# print(type(randidx))    # <class 'numpy.ndarray'>

x_augumented = x_train[randidx].copy()  # .copy() 
y_augumented = y_train[randidx].copy() 
# print(x_augumented.shape)   # (40000, 28, 28)
# print(y_augumented.shape)   # (40000,)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1], x_test.shape[2],1)

x_augumented = x_augumented.reshape(x_augumented.shape[0],
                                    x_augumented.shape[1], 
                                    x_augumented.shape[2],1)
import time
start_time = time.time()
print('시작')
# 변환
x_augumented = train_datagen.flow(x_augumented, y_augumented,
                                  batch_size=augument_size,
                                  save_to_dir='d:/study_data/_temp/',
                                  shuffle=False).next()[0]  # shuffle=False 하는 이유 randidx로 랜덤으로 가져왔기 때문.
end_time = time.time() - start_time

print(augument_size, "개 걸린시간 : ", round(end_time,3), "초")

# 40000 걸린시간 :  18.454 초
# print(x_augumented)
# print(x_augumented.shape)   (40000, 28, 28, 1)

x_train = np.concatenate((x_train, x_augumented))
y_train = np.concatenate((y_train, y_augumented))
print(x_train.shape, y_train.shape) # (100000, 28, 28, 1) (100000,)



