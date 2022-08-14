import numpy as np


aaa = np.array([-10,2,3,4,5,6,700,8,9,10,11,12,50])
aaa= aaa.reshape(-1,1)
print(aaa.shape)

from sklearn.covariance import EllipticEnvelope
outliers = EllipticEnvelope(contamination=.1)               # .1  = 10%를 넘는 이상치 .2  = 20%를 넘는 이상치 

outliers.fit(aaa)
results =outliers.predict(aaa)
print(results)








# def outliers(data_out) :
#     quartile_1, q2, quartile_3 = np.percentile(data_out,
#                                                [25,50,75])
#     print('1사분위 :',quartile_1)               # 1사분위 : 4.0
#     print('q2:',q2)                             # q2: 7.0  (중위값)
#     print('3사분위 :',quartile_3)               # 3사분위 : 10.0
#     iqr = quartile_3 - quartile_1
#     print('iqr:', iqr)                          # iqr: 6.0
#     lower_bound = quartile_1 -(iqr *1.5)
#     upper_bound = quartile_3 +(iqr *1.5)
#     print('upper_bound:',upper_bound)           # upper_bound: 19.0
#     print('lower_bound:',lower_bound)           # lower_bound: -5.0
#     return np.where((data_out>upper_bound)|     # aaa에서 upper_bound(19.0)보다 높은 번째 수 8
#                     (data_out<lower_bound))     # aaa에서 lower_bound(-5.0)보다 높은 번째 수 2
  
# outliers_loc = outliers(aaa)
# print('이상치의 위치:',outliers_loc)            # 이상치의 위치: (array([0, 12], dtype=int64),
    
# import matplotlib.pyplot as plt
# plt.boxplot(aaa)