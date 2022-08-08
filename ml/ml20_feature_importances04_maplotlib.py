import sklearn as datasets
from sklearn.datasets import load_diabetes
import numpy as np

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
           train_size=0.8, shuffle=True, random_state=123)

#2. 모델구성
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

# model = DecisionTreeRegressor()
# model = RandomForestRegressor()
model = GradientBoostingRegressor()
# model = XGBRegressor()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)
print('model.score : ', result)

from sklearn.metrics import r2_score
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('accuracy_score : ', r2)

print(model,':',model.feature_importances_)   # 

import matplotlib.pyplot as plt

def plot_feature_importances(model):
    n_feature = datasets.data.shape[1]
    plt.barh(np.arange(n_feature), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_feature), datasets.feature_names)
    plt.xlabel('Feature Importances')
    plt.ylabel('Feature')
    plt.ylim(-1, n_feature)
    plt.title(model)

plot_feature_importances(model) 
plt.show()   