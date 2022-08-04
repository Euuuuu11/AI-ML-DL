from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense
from sklearn.svm import LinearSVR # 레거시한 리니어 모델 사용


datasets = fetch_california_housing()
x = datasets.data
y = datasets.target


x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.7, shuffle=True, random_state=66)

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
# 결과 : -2.7378798662577166

# SVR
# 결과 : -0.0370628287403556

# LinearRegression
# 결과 : 0.6001949284390906

# KNeighborsRegressor
# 결과 : 0.13188588139050017

# DecisionTreeRegressor
# 결과 : 0.5930609567042704

# RandomForestRegressor
# 결과 : 0.8086986520753949