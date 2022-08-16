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

def remove_outlier(input_data):
    q1 = input_data.quantile(0.25) # 제 1사분위수
    q3 = input_data.quantile(0.75) # 제 3사분위수
    iqr = q3 - q1 # IQR(Interquartile range) 계산
    minimum = q1 - (iqr * 1.5) # IQR 최솟값
    maximum = q3 + (iqr * 1.5) # IQR 최댓값
    ### IQR 범위 내에 있는 데이터만 산출(IQR 범위 밖의 데이터는 이상치) ###
    df_removed_outlier = input_data[(minimum < input_data) & (input_data < maximum)]
    return df_removed_outlier

# 이상치 제거한 데이터셋
data = remove_outlier(data)
print(data[:40])

# 이상치 채워주기
data = data.interpolate()
print(data[:40])

x = data2[:, :11]
y = data2[:, 11]
# print(x.shape, y.shape) # (4898, 11) (4898,)

# print(np.unique(y, return_counts=True))
# (array([3., 4., 5., 6., 7., 8., 9.]), array([  20,  163, 1457, 2198,  880,  175,    5], dtype=int64))
# print(data['quality'].value_counts())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=123, shuffle=True,
                                                    train_size=0.8, stratify=y)

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
