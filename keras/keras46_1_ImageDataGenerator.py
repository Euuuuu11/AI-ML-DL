from matplotlib.pyplot import cla
import numpy as np  
from keras.preprocessing.image import ImageDataGenerator
from sklearn import datasets

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
    'd:/_data/image/brain/train/',
    target_size=(200, 200), # 크기 맞추기
    batch_size=5,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True,
    )   # Found 160 images belonging to 2 classes.
   
xy_test = test_datagen.flow_from_directory(   # directory = 폴더
    'd:/_data/image/brain/test/',
    target_size=(200, 200), # 크기 맞추기
    batch_size=5,
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





