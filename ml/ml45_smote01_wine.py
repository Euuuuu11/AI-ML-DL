# SMOTE 단점 : 데이터가 많으면 속도가 많이 느려진다.

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE    # pip install imblearn
import sklearn as sk
# print('사이킷런 : ', sk.__version__)    # 사이킷런 :  1.1.2

#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets['target']
# print(x.shape, y.shape) # (178, 13) (178,)
# print(type(x))          # <class 'numpy.ndarray'>
# print(np.unique(y, return_counts=True)) # (array([0, 1, 2]), array([59, 71, 48], dtype=int64))
# print(pd.Series(y).value_counts())
# 1    71
# 0    59
# 2    48
# print(y)
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]

x = x[:-25]
y = y[:-25]
# print(pd.Series(y).value_counts())
# 1    71
# 0    59
# 2    8

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=123, shuffle=True, 
                                                    train_size=0.75, stratify=y)

print(pd.Series(y_train).value_counts())
# 1    53
# 0    44
# 2     6

#2. 모델
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

#3. 훈련
model.fit(x_train,y_train)

#4. 평가,예측
y_predict = model.predict(x_test)

from sklearn.metrics import accuracy_score, f1_score    

score = model.score(x_test, y_test)
# print('model.score : ', score)
print('acc_score : ', accuracy_score(y_test,y_predict))
print('f1_score(macro) : ', f1_score(y_test,y_predict, average='macro'))
# print('f1_score(micro) : ', f1_score(y_test,y_predict, average='micro'))

# 기본결과
# acc_score :  0.9777777777777777
# f1_score(macro) :  0.9797235023041475

# 데이터 축소 후(2라벨을 40개 줄인 후)
# acc_score :  0.9428571428571428
# f1_score(macro) :  0.8596176821983273

print('============================ SMOTE 적용 후 ============================')
smote = SMOTE(random_state=123)
x_train, y_train = smote.fit_resample(x_train,y_train)

# print(pd.Series(y_train).value_counts())
# 0    53
# 1    53
# 2    53

#2. 모델, #3. 훈련
model = RandomForestClassifier()
model.fit(x_train, y_train)

#4. 평가,예측
y_predict = model.predict(x_test)

from sklearn.metrics import accuracy_score, f1_score    

score = model.score(x_test, y_test)
# print('model.score : ', score)
print('acc_score : ', accuracy_score(y_test,y_predict))
print('f1_score(macro) : ', f1_score(y_test,y_predict, average='macro'))
# print('f1_score(micro) : ', f1_score(y_test,y_predict, average='micro'))

# acc_score :  0.9743589743589743
# f1_score(macro) :  0.9797235023041475









