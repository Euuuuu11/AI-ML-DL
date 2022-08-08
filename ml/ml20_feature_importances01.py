import numpy as datasets
from sklearn.datasets import load_iris

#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
           train_size=0.8, shuffle=True, random_state=123)

#2. 모델구성
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# model = DecisionTreeClassifier()
# model = RandomForestClassifier()
# model = GradientBoostingClassifier()
model = XGBClassifier()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)
print('model.score : ', result)

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('accuracy_score : ', acc)

print(model,':',model.feature_importances_)   # 전체 feature를 다 사용 안해도 된다.

# DecisionTreeClassifier() : [0.01253395 0.01253395 0.54934776 0.42558435]
# RandomForestClassifier() : [0.10257311 0.02550059 0.43851457 0.43341174]
# GradientBoostingClassifier() : [0.00079275 0.02355941 0.65250973 0.32313811]
# XGBClassifier() : [0.0089478  0.01652037 0.75273126 0.22180054]