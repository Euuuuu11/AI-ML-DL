# 아웃라이어 확인
# 아웃라이어 처리
# 돌려봐
import pandas as pd
import numpy as np

#1. 데이터
path = 'D:/study_data/_data/'
data = pd.read_csv(path + 'winequality-white.csv', index_col=None,
                   header=0, sep=';')  

# print(data.shape)   # (4898, 12)
# print(data.describe())  # 데이터의 내용들 나옴.
# print(data.info())

# pandas를 numpy로 바꾸는 법
data2 = data.to_numpy()
# data = data.values
# print(type(data))   # <class 'numpy.ndarray'>
# print(data.shape)   # (4898, 12)

x = data2[:, :11]
y = data2[:, 11]

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
    
outliers_loc = outliers(data2)
print('이상치의 위치 : ', outliers_loc)     

x = np.delete(x, outliers_loc, 0)
y = np.delete(y, outliers_loc, 0)

# print(x.shape, y.shape) # (4898, 11) (4898,)

# print(np.unique(y, return_counts=True))
# (array([3., 4., 5., 6., 7., 8., 9.]), array([  20,  163, 1457, 2198,  880,  175,    5], dtype=int64))
# print(data['quality'].value_counts())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=123, shuffle=True,
                                                    train_size=0.8)

#2. 모델
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

#3. 훈련
model.fit(x_train,  y_train)

#4. 평가,예측
y_predict = model.predict(x_test)

from sklearn.metrics import accuracy_score, f1_score    

score = model.score(x_test, y_test)
print('model.score : ', score)
print('acc_score : ', accuracy_score(y_test,y_predict))
print('f1_score(macro) : ', f1_score(y_test,y_predict, average='macro'))
print('f1_score(micro) : ', f1_score(y_test,y_predict, average='micro'))
