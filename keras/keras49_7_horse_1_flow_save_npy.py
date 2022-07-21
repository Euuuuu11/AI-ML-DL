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
    'D:/study_data/_data/horse-or-human/',
    target_size=(150, 150), # 크기 맞추기
    batch_size=527,
    class_mode='binary',
    shuffle=True,
    )
x = xy_data[0][0]
y = xy_data[0][1]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7, random_state=66)

train_datagen = ImageDataGenerator(
    horizontal_flip=True,
    # vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    # shear_range=0.7,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator()

################################### 스케일링 ######################################
from sklearn.preprocessing import MinMaxScaler, StandardScaler
x_train1 = x_train.reshape((x_train.shape[0]), (x_train.shape[1])*(x_train.shape[2])*3)
x_test1 = x_test.reshape((x_test.shape[0]), (x_test.shape[1])*(x_test.shape[2])*3)

scaler = MinMaxScaler()
x_train1 = scaler.fit_transform(x_train1)
x_test1 = scaler.transform(x_test1)

x_train = x_train1.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 3)
x_test = x_test1.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 3)

###################################################################################


augument_size = 40 # 증폭
randindx = np.random.randint(x_train.shape[0], size = augument_size)
# print(randindx,randindx.shape) # (40000,)
# print(np.max(randindx), np.min(randindx)) # 59997 2
# print(type(randindx)) # <class 'numpy.ndarray'>

x_augumented = x_train[randindx].copy()
# print(x_augumented,x_augumented.shape) # (40000, 28, 28, 1)
y_augumented = y_train[randindx].copy()
# print(y_augumented,y_augumented.shape) # (40000,)

# x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
# x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
# x_augumented = x_augumented.reshape(x_augumented.shape[0], 
#                                     x_augumented.shape[1], x_augumented.shape[2], 1)

x_augumented = train_datagen.flow(x_augumented, y_augumented,
                                  batch_size=augument_size,
                                  shuffle=False).next()[0]
# print(x_augumented[0][1])

x_train = np.concatenate((x_train, x_augumented))
y_train = np.concatenate((y_train, y_augumented))



xy_train = test_datagen.flow(x_train, y_train,
                                  batch_size=150,
                                  shuffle=False)


# print((x_train.shape),(y_train.shape))  # (368, 150, 150, 3) (368,)
# # 현재 5,200,200,1 짜리 데이터가 32덩어리

np.save('d:/study_data/_save/_npy/keras49_7_train_x.npy', arr=x_train)
np.save('d:/study_data/_save/_npy/keras49_7_train_y.npy', arr=y_train)
np.save('d:/study_data/_save/_npy/keras49_7_test_x.npy', arr=x_test)
np.save('d:/study_data/_save/_npy/keras49_7_test_y.npy', arr=y_test)