# smote 넣은거 안넣은거 비교
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE    # pip install imblearn
import sklearn as sk
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

x = data.data
y = data.target

# data2 =data.values

# x = data.values[:,0:11]
# y = data.values[:,11]


newlist = []
for i in y :
    if i<=5 :
        newlist += [0]
    elif i<=6:
        newlist += [1]
    else:
        newlist += [2]
        
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

# 데이터 축소 후(2라벨을 25개 줄인 후)
# acc_score :  0.9428571428571428
# f1_score(macro) :  0.8596176821983273
print('============================ SMOTE 적용 후 ============================')
smote = SMOTE(random_state=123,k_neighbors=3)
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

# acc_score :  0.6554552912223134
# f1_score(macro) :  0.41831204455345933


# acc_score :  0.958041958041958
# f1_score(macro) :  0.9553682896379526
# ============================ SMOTE 적용 후 ============================
# acc_score :  0.965034965034965
# f1_score(macro) :  0.9629399263981755