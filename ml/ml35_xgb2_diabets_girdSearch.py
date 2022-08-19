from sklearn.datasets import load_diabetes
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold,StratifiedKFold,train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
import time 

#1. 데이터 
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x.shape,y.shape)          # (569, 30) (569,)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.8, random_state=123)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5

kfold = KFold(n_splits=n_splits ,shuffle=True, random_state=123)

# 'n_estimators':[100, 200, 300, 400, 500, 1000]}                 # 디폴트 100 /1-무한대/ 정수
# 'learning_rate':[0.1, 0.2, 0.3, 0.5, 1, 0.01, 0.001]            # 디폴트 0.3/ 0-1 / 다른이름:eta
# 'max_depth':[None, 2, 3, 4, 5, 6, 7, 8, 9, 10]                  # 디폴트 6  /0-무한대 /낮을수록 좋은편./정수
# 'gamma': [0, 1, 2, 3, 4, 5, 7, 10 ,100]                         # 디폴트 0  / 0-무한대 / # loss값을 조각낸다.
# 'min_child_weight':[0,0.1,0.001,1,2,3,4,5,6,10,100]             # 디폴트 1 / 0-무한대 
# 'subsample':[0,0.1,0.2,0.3,0.5,0.7,1]                           # 디폴트 1 / 0-1 데이터에 일정량을 샘플로 쓰겠다.
# 'colsample_bytree':[0,0.1,0.2,0.3,0.5,0.7,1]}                   # 디폴트 1 / 0-1
#'colsample_bylevel':[0,0.1,0.2,0.3,0.5,0.7,1]}                   # 디폴트 1 / 0-1
#'colsample_byload':[0,0.1,0.2,0.3,0.5,0.7,1]}                    # 디폴트 1 / 0-1
# 'reg_alpha':[0,0.1,0.01,0.001,1,2,10]                           # 디폴트 0 / 0-무한대 / L1 절대값 가중치 규제 / alpha
# 'reg_lambda':[0,0.1,0.01,0.001,1,2,10]                          # 디폴트 1 / 0-무한대 / L2 제곱값 가중치 규제 / lambda

parameters = {'n_estimators':[100, 200, 300, 400, 500, 1000],
              'learning_rate':[0.1, 0.2, 0.3, 0.5, 1, 0.01, 0.001],
              'max_depth':[None, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              'gamma': [0, 1, 2, 3, 4, 5, 7, 10 ,100],
              'min_child_weight':[0,0.1,0.001,1,2,3,4,5,6,10,100],
              'subsample':[0,0.1,0.2,0.3,0.5,0.7,1] ,
              'colsample_bytree':[0,0.1,0.2,0.3,0.5,0.7,1],
              'colsample_bylevel':[0,0.1,0.2,0.3,0.5,0.7,1],
            #   'colsample_byload':[0,0.1,0.2,0.3,0.5,0.7,1],
              'reg_alpha':[0,0.1,0.01,0.001,1,2,10],
              'reg_lambda':[0,0.1,0.01,0.001,1,2,10]
              }  


#2. 모델 

xgb =XGBRegressor(random_state=123,
                    )

model = GridSearchCV(xgb, parameters, cv =kfold, n_jobs=8)

import time
start = time.time()
model.fit(x_train,y_train)
end = time.time()

print('최상의매개변수 : ',model.best_params_)
print('최상의 점수 :', model.best_score_)

#4. 평가 예측

results= model.score(x_test,y_test)
print("결과 :",results)
print("시간 :", end-start )

# 'n_estimators'
# 최상의매개변수 :  {'n_estimators': 100}
# 최상의 점수 : 0.9626373626373628


# 'learning_rate'
# 최상의매개변수 :  {'learning_rate': 0.01, 'n_estimators': 1000}
# 최상의 점수 : 0.9670329670329672

# 'max_depth'
# 최상의매개변수 :  {'learning_rate': 0.1, 'max_depth': 2, 'n_estimators': 1000}
# 최상의 점수 : 0.9692307692307693

# 'gamma'
# 최상의매개변수 :  {'gamma': 1, 'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 1000}
# 최상의 점수 : 0.9736263736263737

# 'min_child_weight'
# 최상의매개변수 :  {'gamma': 1, 'learning_rate': 0.1, 'max_depth': 3, 'min_child_weight': 1, 'n_estimators': 1000}
# 최상의 점수 : 0.9736263736263737

# 'subsample'
# 최상의매개변수 :  {'gamma': 1, 'learning_rate': 0.1, 'max_depth': 3, 'min_child_weight': 1, 'n_estimators': 1000, 'subsample': 1}
# 최상의 점수 : 0.9736263736263737

# 'colsample_bytree
# 최상의매개변수 :  {'colsample_bytree': 0.5, 'gamma': 1, 'learning_rate': 0.1, 'max_depth': 3, 'min_child_weight': 1, 'n_estimators': 1000, 'subsample': 1}
# 최상의 점수 : 0.9736263736263737

# # 'colsample_bylevel
# 상의매개변수 :  {'colsample_bylevel': 1, 'gamma': 1, 'learning_rate': 0.1, 'max_depth': 3, 'min_child_weight': 1, 'n_estimators': 1000, 'subsample': 1}
# 최상의 점수 : 0.9736263736263737

# 'colsample_byload
# 최상의매개변수 :  {'colsample_byload': 0.5, 'gamma': 1, 'learning_rate': 0.1, 'max_depth': 3, 'min_child_weight': 1, 'n_estimators': 1000, 'subsample': 1}
# 최상의 점수 : 0.9736263736263737