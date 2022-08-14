import numpy as np

aaa = np.array([[-10,2,3,4,5,6,7,8,9,10,11,12,50],
               [100,200,-30,400,500,600,-70000,800,900,1000,210,420,350]])

aaa1 = aaa[0]
aaa2 = aaa[1]
print(aaa1)
print(aaa2)



aaa1= aaa1.reshape(-1,1)
aaa2= aaa2.reshape(-1,1)

print(aaa)
print(aaa.shape)
exit()
from sklearn.covariance import EllipticEnvelope
outliers = EllipticEnvelope(contamination=.1)               # .1  = 10%를 넘는 이상치 .2  = 20%를 넘는 이상치 

outliers.fit(aaa1)
outliers.fit(aaa2)

results =outliers.predict(aaa1)
results2 =outliers.predict(aaa2)

print(results)
print(results2)


# import numpy as np 
# aaa = np.array([[-10,2,3,4,5,6,7,8,9,10,11,12,50], 
#                [100,200,-30,400,500,600,-70000,800,900,1000,510,420,350]])


# aaa = np.transpose(aaa)
# print(aaa)
# print(aaa.shape)


# def outliers(data_out):
#     quartile_1, q2, quartile_3 = np.percentile(data_out,[25,50,75])
#     print('1사분위 : ', quartile_1)
#     print('q2 : ', q2)
#     print('3사분위 : ', quartile_3)
#     iqr = quartile_3 - quartile_1
#     print('iqr : ,', iqr)
#     lower_bound = quartile_1 - (iqr *1.5)
#     upper_bound = quartile_3 + (iqr *1.5)
#     return np.where((data_out>upper_bound)|(data_out<lower_bound))
    
# outliers_loc1 = outliers(aaa[:,0])
# outliers_loc2 = outliers(aaa[:,1])
# print('이상치의 위치 : ', outliers_loc1)
# print('이상치의 위치 : ', outliers_loc2)

# import matplotlib.pyplot as plt 
# plt.boxplot(aaa)
# plt.show()