from unittest import result
import numpy as np
from sklearn.datasets import load_boston, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
# import sklearn as sk
# print(sk.__version__)
import warnings
warnings.filterwarnings(action='ignore')

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target
# print(x.shape, y.shape) # (506, 13) (506,)

pca = PCA(n_components=12)   # 주성분 분석, 차원축소(차원 = 컬럼)
x = pca.fit_transform(x)
print(x.shape) # (506, 12)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                train_size=0.8, random_state=123, shuffle=True)

#2.모델
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

model = RandomForestRegressor()

#3. 훈련
model.fit(x_train, y_train) #, eval_metric='error')

#4. 평가
result = model.score(x_test,y_test)
print('결과 : ', result)

# (506, 13)
# 결과 :  0.7602148280611554

# (506, 11)
# 결과 :  0.8021570204121178

