import numpy as np
from sklearn.decomposition import PCA
from keras.datasets import mnist
(x_train, _),(x_test, _)= mnist.load_data()

# print(x_train.shape, x_test.shape)  # (60000, 28, 28) (10000, 28, 28)

x = np.append(x_train, x_test, axis = 0)
# print(x.shape)    # (70000, 28, 28)
###################################################################
# 실습
# pca를 통해 0.95 이상인 n_components는 몇개 ?
# 0.95
# 0.99
# 0.999
# 1.0
# 힌드 np.argmax
##################################################################
x = x.reshape(70000, 28*28)
# print(x.shape) #  (70000, 784)

pca = PCA(n_components=784)   
x = pca.fit_transform(x)












