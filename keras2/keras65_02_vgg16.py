import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16

# model = VGG16()    # include_top=True, input_shape=(224, 224, 3)
vgg16 = VGG16(weights='imagenet', include_top=False,
              input_shape=(32, 32, 3))

# vgg16.summary()
# vgg16.trainable=False         # 가중치 동결시킴.
# vgg16.summary()

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10))

model.trainable = False

model.summary()
                                # Trainable:True / VGG False / model False
print(len(model.weights))           # 30 / 30 / 30
print(len(model.trainable_weights)) # 30 / 4  / 0




