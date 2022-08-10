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

pca_EVR = pca.explained_variance_ratio_
cumsum = np.cumsum(pca_EVR) 
print(cumsum)

print(np.argmax(cumsum >= 0.95) + 1)    # 154
print(np.argmax(cumsum >= 0.99) + 1)    # 331
print(np.argmax(cumsum >= 0.999) + 1)   # 486
print(np.argmax(cumsum >= 1.0) + 1)     # 713

# import matplotlib.pyplot as plt
# plt.plot(cumsum)
# plt.grid()
# plt.show()

# print(np.argwhere(cumsum>=0.95)[0]) # [153]
# print(np.argwhere(cumsum>=1)[0])    # [712]





