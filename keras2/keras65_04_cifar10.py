# trainable = True, False 만들어서 비교

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import VGG16
import time

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train / 255
x_test = x_test / 255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

vgg16 = VGG16(weights='imagenet', include_top=False,
              input_shape=(32, 32, 3))
# vgg16.trainable = False  # vgg16의 레이어에 대해서는 훈련을 안시킨다.(가중치 동결)

model = Sequential()
model.add(vgg16)
model.add(GlobalAveragePooling2D())
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax'))



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', mode='min', patience=15)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto', verbose=1, factor=0.5)

start = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=128, callbacks=[es, reduce_lr], validation_split=0.2)
end = time.time() - start

loss = model.evaluate(x_test, y_test)
print("걸린 시간 : ", round(end, 2))
print('loss, acc ', loss)

# False일 때
# 걸린 시간 :  198.23
# loss, acc  [1.1391757726669312, 0.6126000285148621]

# True일 때
# 걸린 시간 :  290.19
# loss, acc  [1.4466814994812012, 0.8065999746322632]

