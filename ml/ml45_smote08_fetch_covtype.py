# 증폭 후 저장
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE    # pip install imblearn
import sklearn as sk
from sklearn.datasets import fetch_covtype

data = fetch_covtype()

x = data.data
y = data.target

# data2 =data.values

# x = data.values[:,0:11]
# y = data.values[:,11]

print(pd.Series(y).value_counts())


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=123, shuffle=True, 
                                                    train_size=0.75, stratify=y)

print(pd.Series(y_train).value_counts())
# 6.0    1648
# 5.0    1093
# 7.0     660
# 8.0     131
# 4.0     122
# 3.0      15
# 9.0       4

#2. 모델
# from sklearn.ensemble import RandomForestRegressor
# model = RandomForestRegressor()

# path = 'D:\study_data\_save/_xg/'

# # save /pickle 양대산맥
# import joblib
# joblib.dump(model,(path+'ml45_fetch_covtype_save.dat'))

import joblib
path = 'D:/study_data/_save/_xg/'
model = joblib.load(path + 'ml45_smote_save.dat')

#3. 훈련
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(x_train,y_train)

#4. 평가,예측
y_predict = model.predict(x_test)

from sklearn.metrics import accuracy_score, f1_score    

score = model.score(x_test, y_test)
# print('model.score : ', score)
print('acc_score : ', accuracy_score(y_test, y_predict))
print('f1_score(macro) : ', f1_score(y_test,y_predict, average='macro'))
# print('f1_score(micro) : ', f1_score(y_test,y_predict, average='micro'))


# 증폭 전
# acc_score :  0.9539424314816217
# f1_score(macro) :  0.9241513178102323

# 증폭 후
# acc_score :  0.9541351985845387
# f1_score(macro) :  0.9225708873716857