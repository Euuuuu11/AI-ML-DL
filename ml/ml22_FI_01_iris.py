# 실습 
# 피처임포턴스가 전체 중요도애서 하위 20~25% 컬럼들을 제거하여
# 데이터셋 재구성 후
# 각 모델별로 돌려서 결과 도출

# 기존 모델결과와 비교

#2. 모델구성
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

# 결과비교
# 1. DecisionTreeClassifier
# 기존 acc
# 컬럼 삭제 후 acc


import sklearn as datasets
from sklearn.datasets import load_iris
import numpy as np

#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target
# print(x.shape)

x = x.drop(x.columns[[3]], axis=1)
# x = np.delete(x, 1, axis=1)
# print(x.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
           train_size=0.8, shuffle=True, random_state=123)

#2. 모델구성
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

# model = DecisionTreeRegressor()
model = RandomForestRegressor()
# model = GradientBoostingRegressor()
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

print(model,':',model.feature_importances_)   # 전체 feature를 다 사용 안해도 된다.

# model.score :  0.13165496230674723
# DecisionTreeRegressor() : [0.08757925 0.01965106 0.22933341 0.05826787 0.05153937 0.06763122
#  0.03751957 0.0108828  0.36260137 0.07499408]

# model.score :  0.5242156421479274 
# RandomForestRegressor() : [0.06058436 0.01080491 0.30027134 0.09801928 0.03990439 0.05543511
#  0.05432961 0.03089632 0.26858025 0.08117443]

# model.score :  0.5565236504321536
# GradientBoostingRegressor() : [0.04944464 0.01077472 0.30289721 0.11204719 0.02766884 0.05476751
#  0.03950266 0.01940299 0.33886375 0.04463048]

# model.score :  0.4590400803596264
# XGBClassifier() : [0.03234756 0.0447546  0.21775807 0.08212128 0.04737141 0.04843819
#  0.06012432 0.09595273 0.30483875 0.06629313]