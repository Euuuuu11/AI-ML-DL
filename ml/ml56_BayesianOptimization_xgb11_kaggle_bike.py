from bayes_opt import BayesianOptimization
from lightgbm import LGBMRegressor, LGBMClassifier
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import warnings
warnings.filterwarnings('ignore')
from xgboost import XGBClassifier, XGBRegressor

#1.데이터
path = 'C:\study\_data\kaggle_bike/'
train_set = pd.read_csv(path + 'train.csv') # + 명령어는 문자를 앞문자와 더해줌  index_col=n n번째 컬럼을 인덱스로 인식
            
test_set = pd.read_csv(path + 'test.csv') # 예측에서 쓸거임        

###########이상치 처리##############
def dr_outlier(train_set):
    quartile_1 = train_set.quantile(0.25)
    quartile_3 = train_set.quantile(0.75)
    IQR = quartile_3 - quartile_1
    condition = (train_set < (quartile_1 - 1.5 * IQR)) | (train_set > (quartile_3 + 1.5 * IQR))
    condition = condition.any(axis=1)

    return train_set, train_set.drop(train_set.index, axis=0)

dr_outlier(train_set)
####################################


######## 년, 월 ,일 ,시간 분리 ############

train_set["hour"] = [t.hour for t in pd.DatetimeIndex(train_set.datetime)]
train_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(train_set.datetime)]
train_set["month"] = [t.month for t in pd.DatetimeIndex(train_set.datetime)]
train_set['year'] = [t.year for t in pd.DatetimeIndex(train_set.datetime)]
train_set['year'] = train_set['year'].map({2011:0, 2012:1})

test_set["hour"] = [t.hour for t in pd.DatetimeIndex(test_set.datetime)]
test_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(test_set.datetime)]
test_set["month"] = [t.month for t in pd.DatetimeIndex(test_set.datetime)]
test_set['year'] = [t.year for t in pd.DatetimeIndex(test_set.datetime)]
test_set['year'] = test_set['year'].map({2011:0, 2012:1})

train_set.drop('datetime',axis=1,inplace=True) # 트레인 세트에서 데이트타임 드랍
train_set.drop('casual',axis=1,inplace=True) # 트레인 세트에서 캐주얼 레지스터드 드랍
train_set.drop('registered',axis=1,inplace=True)

test_set.drop('datetime',axis=1,inplace=True) # 트레인 세트에서 데이트타임 드랍

# print(train_set)# [10886 rows x 13 columns]
# print(test_set)# [6493 rows x 12 columns]

##########################################


x = train_set.drop(['count'], axis=1)  # drop 데이터에서 ''사이 값 빼기
# print(x)
# print(x.columns)
# print(x.shape) # (10886, 12)
y = train_set['count'] 

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=1234, train_size=0.8
)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
bayesian_params = {      
   'max_depth':(7,13),
    'min_child_weight' : (28,34),
    'subsample':(0.1,4),
    'colsample_bytree':(0.1,3.5),
    'reg_lambda' : (32,38),
    'reg_alpha':(36,39)
}

# def lgb_hamsu(max_depth, min_child_weight,
#               subsample, colsample_bytree, reg_alpha, reg_lambda ):
#     params = {
#         'n_estimators':500, "learning_rate":0.02,
#         'max_depth' : int(round(max_depth)),                    # 무조건 정수형
#         # 'num_leaves' : int(round(num_leaves)),
#         # 'min_child_samples' : int(round(min_child_samples)),
#         'min_child_weight' : int(round(min_child_weight)),      
#         'subsample' : max(min(subsample, 1), 0.5),                # 0~1 사이의 값이 들어오
#         'colsample_bytree' : max(min(colsample_bytree, 1), 0),  
#         # 'max_bin' : max(int(round(max_bin)),8),                # 무조건 10이상 정수형
#         'reg_lambda' : int(round(reg_lambda,0)),                # 무조건 양수만
#         'reg_alpha' : int(round(reg_alpha,0))
        
#     }
    
#     # *여러개의인자를받겠다
#     # **키워드받겠다(딕셔너리형태)
#     model = XGBRegressor(**params)
    
#     model.fit(x_train, y_train,
#               eval_set=[(x_train, y_train), (x_test, y_test)],
#               eval_metric='rmse',
#               verbose=0,
#               early_stopping_rounds=50
#               )
    
#     y_predict = model.predict(x_test)
#     results = r2_score(y_test, y_predict)
    
#     return results

# lgb_bo = BayesianOptimization(f=lgb_hamsu,
#                               pbounds= bayesian_params,
#                               random_state=123
#                               )
# lgb_bo.maximize(init_points=5, n_iter=100)

# print(lgb_bo.max)


########################### [실습] ###########################
#1. 수정한 파라미터로 모델 만들어서 비교!!!

# {'target': 0.8990628525352931, 'params': {'colsample_bytree': 0.5163070271476488, 'max_depth': 9.819095208997838, 'min_child_weight': 31.552553161785713, 
# 'reg_alpha': 33.93391972418055, 'reg_lambda': 35.50813676981957, 'subsample': 0.8551416812982002}}

#2. 수정한 파라미터로 이용해서 재조정!!!!
# {'target': 0.9593712029458259, 'params': {'colsample_bytree': 3.5, 'max_depth': 13.0, 'min_child_weight': 28.0, 
# 'reg_alpha': 36.0, 'reg_lambda': 32.0, 'subsample': 4.0}}



model = XGBRegressor(n_estimators = 500, learning_rate= 0.02, colsample_bytree =max(min(3.5,1),0) ,
max_depth=int(round(13)), min_child_weight =int(round(28)),
reg_alpha= max(36,0), reg_lambda=max( 32,0), subsample=max(min(4,1),0))

model.fit(x_train, y_train)
y_pred = model.predict(x_test)
score = r2_score(y_test, y_pred)
print('파마리터 수정 후 score : ', score)

# 파마리터 수정 후 score : 0.9593712029458259
