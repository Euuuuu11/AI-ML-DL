from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

# 결과비교
# 1. DecisionTreeClassifier
# 기존 acc
# 컬럼 삭제 후 acc


import sklearn as datasets
from sklearn.datasets import load_breast_cancer
import numpy as np

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
# print(x.shape)

# x = x.drop(x.columns[[3]], axis=1)
x = np.delete(x, [10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29], axis=1)
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

# DecisionTreeClassifierr_score :  0.9122807017543859
# RandomForestClassifier_score :  0.9649122807017544
# GradientBoostingClassifier_score :  0.9473684210526315
# XGBClassifier_score :  0.9298245614035088