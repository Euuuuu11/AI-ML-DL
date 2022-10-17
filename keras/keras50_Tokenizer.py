from keras.preprocessing.text import Tokenizer

text = '나는 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다.'    

token = Tokenizer()
token.fit_on_texts([text])

# print(token.word_index)
# {'마구': 1, '매우': 2, '나는': 3, '진짜': 4, '맛있는': 5, '밥을': 6, '엄청': 7, '먹었다': 8}
import numpy as np
x = token.texts_to_sequences([text])
# print(x)
# [[3, 4, 2, 2, 5, 6, 7, 1, 1, 1, 8]]
x = np.array(x)
# [[3 4 2 2 5 6 7 1 1 1 8]]
# print(x)
x= x.reshape(11,1)
# print(x)


from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.preprocessing import OneHotEncoder

# x = to_categorical(x)
# print(x)
# print(x.shape)

# [[[0. 0. 0. 1. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 1. 0. 0. 0. 0.]
#   [0. 0. 1. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 1. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 1. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 1. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 1. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 1.]]]
# (1, 11, 9)


ohe = OneHotEncoder(sparse=False)
x = ohe.fit_transform(x)
print(x)

# [[0. 0. 1. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 1. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 1. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 1. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 1.]]
# print(x.shape)

ohe = OneHotEncoder(sparse=True)
x = ohe.fit_transform(x)
print(x)
#   (7, 14)       1.0
#   (8, 1)        1.0
#   (8, 2)        1.0
#   (8, 4)        1.0
#   (8, 6)        1.0
#   (8, 8)        1.0
#   (8, 10)       1.0
#   (8, 12)       1.0
#   (8, 14)       1.0
#   (9, 1)        1.0
#   (9, 2)        1.0
#   (9, 4)        1.0
#   (9, 6)        1.0
#   (9, 8)        1.0
#   (9, 10)       1.0
#   (9, 12)       1.0
#   (9, 14)       1.0
#   (10, 0)       1.0
#   (10, 2)       1.0
#   (10, 4)       1.0
#   (10, 6)       1.0
#   (10, 8)       1.0
#   (10, 10)      1.0
#   (10, 12)      1.0
#   (10, 15)      1.0


