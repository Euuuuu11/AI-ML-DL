from unittest import result
import numpy as np
from sklearn.datasets import load_breast_cancer, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
# import sklearn as sk
# print(sk.__version__)
import warnings
warnings.filterwarnings(action='ignore')

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
# print(x.shape, y.shape) # (569, 30) (569,)

pca = PCA(n_components=29)   # 주성분 분석, 차원축소(차원 = 컬럼)
x = pca.fit_transform(x)
print(x.shape) # (506, 2)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                train_size=0.8, random_state=123, shuffle=True)

#2.모델
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor

model = RandomForestClassifier()

#3. 훈련
model.fit(x_train, y_train) #, eval_metric='error')

#4. 평가
result = model.score(x_test,y_test)
print('결과 : ', result)

# (569, 30)
# 결과 :  0.9912280701754386

# (569, 4)
# 결과 :  0.9912280701754386