from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score
import numpy as np
from sklearn.svm import LinearSVR # 레거시한 리니어 모델 사용


#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target


x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.8, shuffle=True, random_state=66)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

#2. 모델구성
from sklearn.svm import LinearSVR, SVR
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression, LinearRegression # LinearRegression 회귀 
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
model = RandomForestRegressor()

#3.4. 컴파일, 훈련, 평가, 예측
# scores = cross_val_score(model, x, y, cv=kfold)
scores = cross_val_score(model, x_train, y_train, cv=5)         # 둘다 가능
 
print('R2 : ', scores, '\n cross_val_score : ', round(np.mean(scores), 4))

y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
print(y_predict)


print(y_test)
R2 = r2_score(y_test, y_predict)
print('cross_val_predict R2 : ', R2)

# cross_val_predict R2 :  0.8125660105726229