import sklearn as datasets
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd

#1. 데이터
path = './_data/ddarung/'
train_set = pd.read_csv(path + 'train.csv', 
                        index_col=0) 
test_set = pd.read_csv(path + 'test.csv', 
                       index_col=0)

train_set =  train_set.dropna()
test_set = test_set.fillna(test_set.mean())

x = train_set.drop(['count'], axis=1) 
y = train_set['count']




from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
           train_size=0.8, shuffle=True, random_state=123)

#2. 모델구성
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

model1 = DecisionTreeRegressor()
model2 = RandomForestRegressor()
model3 = GradientBoostingRegressor()
model4 = XGBRegressor()

#3. 훈련
model1.fit(x_train, y_train)
model2.fit(x_train, y_train)
model3.fit(x_train, y_train)
model4.fit(x_train, y_train)

#4. 평가, 예측
# result = model.score(x_test, y_test)
# print('model.score : ', result)

from sklearn.metrics import r2_score
y_predict = model1.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('accuracy_score : ', r2)

print(model1,':',model1.feature_importances_)   # 

import matplotlib.pyplot as plt

def plot_feature_importances(model):
    n_feature = datasets.data.shape[1]
    plt.barh(np.arange(n_feature), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_feature), datasets.feature_names)
    plt.xlabel('Feature Importances')
    plt.ylabel('Feature')
    plt.ylim(-1, n_feature)
    plt.title(model)

model5 ='XGBRegressor()'

plt.subplot(2, 2, 1)
plot_feature_importances(model1) 
plt.title(model1)

plt.subplot(2, 2, 2)
plot_feature_importances(model2) 
plt.title(model2)

plt.subplot(2, 2, 3)
plot_feature_importances(model3) 
plt.title(model3)

plt.subplot(2, 2, 4)
plot_feature_importances(model4) 
plt.title(model5)


plt.show()   
