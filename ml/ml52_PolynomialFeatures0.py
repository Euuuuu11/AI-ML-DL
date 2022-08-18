import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures    # 단항을 다항으로 증폭시킨다.

x = np.arange(8).reshape(4, 2)

# print(x)
# [[0 1]
#  [2 3]
#  [4 5]
#  [6 7]]
# print(x.shape)  # (4, 2)

pf = PolynomialFeatures(degree=2)
x_pf = pf.fit_transform(x)

print(x_pf)
# [[ 1.  0.  1.  0.  0.  1.]
#  [ 1.  2.  3.  4.  6.  9.]
#  [ 1.  4.  5. 16. 20. 25.]
#  [ 1.  6.  7. 36. 42. 49.]]
# print(x_pf.shape)   # (4, 6)

##################################################################################
x = np.arange(12).reshape(4, 3)

print(x)
# [[ 0  1  2]
#  [ 3  4  5]
#  [ 6  7  8]
#  [ 9 10 11]]
print(x.shape)  # (4, 3)

pf = PolynomialFeatures(degree=2)
x_pf = pf.fit_transform(x)

print(x_pf)
# [[  1.   0.   1.   2.   0.   0.   0.   1.   2.   4.]
#  [  1.   3.   4.   5.   9.  12.  15.  16.  20.  25.]
#  [  1.   6.   7.   8.  36.  42.  48.  49.  56.  64.]
#  [  1.   9.  10.  11.  81.  90.  99. 100. 110. 121.]]
print(x_pf.shape)   # (4, 10)

#  [  1.   3.   4.   5.   9.  12.  15.  16.  20.  25.]
# 1은고정   3   4   5  3제곱  3*4   3*5 4제곱  4*5 5의제곱

#################################################################################

x = np.arange(8).reshape(4, 2)

print(x)
# [[0 1]
#  [2 3]
#  [4 5]
#  [6 7]]
print(x.shape)  # (4, 2)

pf = PolynomialFeatures(degree=3)
x_pf = pf.fit_transform(x)

print(x_pf)
# [[  1.   0.   1.   0.   0.   1.   0.   0.   0.   1.]
#  [  1.   2.   3.   4.   6.   9.   8.  12.  18.  27.]
#  [  1.   4.   5.  16.  20.  25.  64.  80. 100. 125.]
#  [  1.   6.   7.  36.  42.  49. 216. 252. 294. 343.]]
print(x_pf.shape)   # (4, 10) 

#  [  1.   2.   3.   4.   6.   9.   8.  12.  18.  27.]
# 
