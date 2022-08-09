# 실습 
# 피처임포턴스가 전체 중요도애서 하위 20~25% 컬럼들을 제거하여
# 데이터셋 재구성 후
# 각 모델별로 돌려서 결과 도출

# 기존 모델결과와 비교

#2. 모델구성
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

# 결과비교
# 1. DecisionTreeClassifier
# 기존 acc
# 컬럼 삭제 후 acc


import sklearn as datasets
from sklearn.datasets import load_iris
import numpy as np

#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target
# print(x.shape)

# x = x.drop(x.columns[[3]], axis=1)
x = np.delete(x, 2, axis=1)
# print(x.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
           train_size=0.8, shuffle=True, random_state=123)

#2. 모델구성
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

model1 = DecisionTreeClassifier()
model2 = RandomForestClassifier()
model3 = GradientBoostingClassifier()
model4 = XGBClassifier()

#3. 훈련
model1.fit(x_train, y_train)
model2.fit(x_train, y_train)
model3.fit(x_train, y_train)
model4.fit(x_train, y_train)


#4. 평가, 예측
result = model1.score(x_test, y_test)
print('model.score : ', result)

from sklearn.metrics import accuracy_score

y_predict = model1.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('DecisionTreeClassifierr_score : ', acc)

print(model1,':',model1.feature_importances_)   # 전체 feature를 다 사용 안해도 된다.

y_predict = model2.predict(x_test)
acc2 = accuracy_score(y_test, y_predict)
print(model2,':',model2.feature_importances_) 

y_predict = model3.predict(x_test)
acc3 = accuracy_score(y_test, y_predict)
print(model3,':',model3.feature_importances_) 

y_predict = model4.predict(x_test)
acc4 = accuracy_score(y_test, y_predict)
print(model4,':',model4.feature_importances_) 


print('DecisionTreeClassifierr_score : ', acc)
print('RandomForestClassifier_score : ', acc2)
print('GradientBoostingClassifier_score : ', acc3)
print('XGBClassifier_score : ', acc4)
# model.score :  0.13165496230674723
# DecisionTreeRegressor() : [0.08757925 0.01965106 0.22933341 0.05826787 0.05153937 0.06763122
#  0.03751957 0.0108828  0.36260137 0.07499408]

# model.score :  0.5242156421479274 
# RandomForestRegressor() : [0.06058436 0.01080491 0.30027134 0.09801928 0.03990439 0.05543511
#  0.05432961 0.03089632 0.26858025 0.08117443]

# model.score :  0.5565236504321536
# GradientBoostingRegressor() : [0.04944464 0.01077472 0.30289721 0.11204719 0.02766884 0.05476751
#  0.03950266 0.01940299 0.33886375 0.04463048]

# model.score :  0.4590400803596264
# XGBClassifier() : [0.03234756 0.0447546  0.21775807 0.08212128 0.04737141 0.04843819
#  0.06012432 0.09595273 0.30483875 0.06629313]


# DecisionTreeClassifierr_score :  0.9666666666666667
# RandomForestClassifier_score :  0.9666666666666667
# GradientBoostingClassifier_score :  0.9666666666666667
# XGBClassifier_score :  0.9666666666666667