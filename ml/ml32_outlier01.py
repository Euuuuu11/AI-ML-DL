import numpy as np
aaa = np.array([1, 2, -30, 4, 5, 6, 7, 8, 900, 10])

def outliers(data_out) :
    quartile_1, q2, quartile_3 = np.percentile(data_out,
                                               [25, 50, 75])
    print('1사분위 : ',quartile_1)
    print('q2 : ', q2)
    print('3사분위 : ',quartile_3)
    iqr = quartile_3 - quartile_1
    print('iqr : ', iqr)
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data_out>upper_bound) |
                    (data_out<lower_bound))
    
outliers_loc = outliers(aaa)
print('이상치의 위치 : ', outliers_loc)     
                
import matplotlib.pyplot as plt
plt.boxplot(aaa)
plt.show()

