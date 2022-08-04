from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.svm import LinearSVR # 레거시한 리니어 모델 사용


#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target


x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.8, shuffle=True, random_state=66)

#2. 모델구성
from sklearn.svm import LinearSVR, SVR
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression, LinearRegression # LinearRegression 회귀 
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
model = RandomForestRegressor()


#3. 컴파일, 훈련
model.fit(x_train, y_train)


#4. 평가, 예측
result = model.score(x_test, y_test) # evaluate 대신 score 사용
print('결과 :', result)

# LinearSVR 결과
# 결과 : 0.7456066158870449

# SVR
# 결과 : 0.23474677555722312

# LinearRegression
# 결과 : 0.8111288663608656

# KNeighborsRegressor
# 결과 : 0.5900872726222293

# DecisionTreeRegressor
# 결과 : 0.7993703473269131

# RandomForestRegressor
# 결과 : 0.9269482382595485