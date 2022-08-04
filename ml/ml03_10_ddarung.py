#데이콘 따릉이 문제풀이
import numpy as np
import pandas as pd
from sklearn import metrics
from tensorflow.python.keras.models import Sequential,  load_model
from tensorflow.python.keras.layers import Activation, Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout,LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

#1. 데이터
path = './_data/ddarung/'
train_set = pd.read_csv(path + 'train.csv', 
                        index_col=0) 


test_set = pd.read_csv(path + 'test.csv', 
                       index_col=0)



train_set =  train_set.dropna()

test_set = test_set.fillna(test_set.mean())


x = train_set.drop(['count'], axis=1) 
#

y = train_set['count']



x_train, x_test, y_train, y_test = train_test_split(x,y,
             train_size=0.75, shuffle=True, random_state=85)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)



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

# LinearSVR
# 결과 : 0.5109276183781397

# SVR
# 결과 : 0.4153046602834609

# LinearRegression
# 결과 : 0.5879603377840328

# KNeighborsRegressor
# 결과 : 0.638962207620777

# DecisionTreeRegressor
# 결과 : 0.48593090572171027

# RandomForestRegressor
# 결과 : 0.7488049910765501