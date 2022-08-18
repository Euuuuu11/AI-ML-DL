import numpy as np
import pandas as pd

from sklearn.ensemble import VotingClassifier, VotingRegressor   # 투표를 통해 최종 예측 결과를 결정하는 방식
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
#1. 데이터
datasets = fetch_california_housing()

df = pd.DataFrame(datasets.data, columns=datasets.feature_names)
# print(df.head(7))

x_train, x_test, y_train, y_test = train_test_split(
    datasets.data, datasets.target, train_size= 0.8, random_state=123,
    )

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
xg = XGBRegressor()
lg = LGBMRegressor()
cat = CatBoostRegressor(verbose=0)

model = VotingRegressor(
    estimators=[('XG', xg), ('LG', lg), ('CAT', cat)],
           # hard
)
# softvoting = 각 분류기별 레이블 값 결정 확률을 평균 낸 확률이 가장 높은 레이블 값을 최종 보팅 결과값으로 선정
# hardvoting = 예측한 결과값들 중 다수의 분류기가 결정한 예측 결과값을 최종 보팅 결과값으로 선정

#3. 훈련
model.fit(x_train, y_train)

#4. 평가,예측
y_predict = model.predict(x_test)

score = r2_score(y_test, y_predict)
print('voting : ', round(score, 4))

# voting :  0.9912

classifiers = [xg, lg, cat]
for model2 in classifiers:
    model2.fit(x_train, y_train)
    y_predict = model2.predict(x_test)
    score2 = r2_score(y_test,y_predict)
    class_name = model2.__class__.__name__
    print('{0} 정확도 : {1:.4f}'.format(class_name, score2))
    
# voting :  0.853
# XGBRegressor 정확도 : 0.8331
# LGBMRegressor 정확도 : 0.8413
# CatBoostRegressor 정확도 : 0.8571