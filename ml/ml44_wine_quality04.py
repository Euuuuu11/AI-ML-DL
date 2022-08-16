import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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
# print(x.shape, y.shape) # (4898, 11) (4898,)

print(np.unique(y, return_counts=True))
# (array([3., 4., 5., 6., 7., 8., 9.]), array([  20,  163, 1457, 2198,  880,  175,    5], dtype=int64))
print(data['quality'].value_counts())

print(y[:20])
# [6. 6. 6. 6. 6. 6. 6. 6. 6. 6. 5. 5. 5. 7. 5. 7. 6. 8. 6. 5.]

newlist = []
for i in y :
    if i<=7 :
        newlist += [0]
    elif i==8:
        newlist += [1]
    else:
        newlist += [2]
             
print(np.unique(newlist, return_counts=True))   
# (array([0, 1, 2]), array([1640, 2198, 1060], dtype=int64))    


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=123, shuffle=True,
                                                    train_size=0.8, stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

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

# model.score :  0.736734693877551
# acc_score :  0.736734693877551
# f1_score(macro) :  0.44708577437284064
# f1_score(micro) :  0.7367346938775511