from bayes_opt import BayesianOptimization
from lightgbm import LGBMRegressor, LGBMClassifier
import numpy as np

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import warnings
warnings.filterwarnings('ignore')
from xgboost import XGBClassifier, XGBRegressor

#1.데이터
datasets = load_wine()
x, y = datasets.data, datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=1234, train_size=0.8
)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
bayesian_params = {      
    'max_depth':(7,13),
    'min_child_weight' : (0,0.3),
    'subsample':(0.2,3.5),
    'colsample_bytree':(0.2,0.8),
    'reg_lambda' : (22,28),
    'reg_alpha':(8,14)
}

def lgb_hamsu(max_depth, min_child_weight,
              subsample, colsample_bytree, reg_alpha, reg_lambda ):
    params = {
        'n_estimators':500, "learning_rate":0.02,
        'max_depth' : int(round(max_depth)),                    # 무조건 정수형
        # 'num_leaves' : int(round(num_leaves)),
        # 'min_child_samples' : int(round(min_child_samples)),
        'min_child_weight' : int(round(min_child_weight)),      
        'subsample' : max(min(subsample, 1), 0.5),                # 0~1 사이의 값이 들어오
        'colsample_bytree' : max(min(colsample_bytree, 1), 0),  
        # 'max_bin' : max(int(round(max_bin)),8),                # 무조건 10이상 정수형
        'reg_lambda' : int(round(reg_lambda,0)),                # 무조건 양수만
        'reg_alpha' : int(round(reg_alpha,0))
        
    }
    
    # *여러개의인자를받겠다
    # **키워드받겠다(딕셔너리형태)
    model = XGBClassifier(**params)
    
    model.fit(x_train, y_train,
              eval_set=[(x_train, y_train), (x_test, y_test)],
              eval_metric='merror',
              verbose=0,
              early_stopping_rounds=50
              )
    
    y_predict = model.predict(x_test)
    results = accuracy_score(y_test, y_predict)
    
    return results

lgb_bo = BayesianOptimization(f=lgb_hamsu,
                              pbounds= bayesian_params,
                              random_state=123
                              )
lgb_bo.maximize(init_points=5, n_iter=100)

print(lgb_bo.max)


########################### [실습] ###########################
#1. 수정한 파라미터로 모델 만들어서 비교!!!

# {'target': 1.0, 'params': {'colsample_bytree': 0.5, 'max_depth': 10.0, 'min_child_weight': 0.0, 'reg_alpha': 11.661648294341814,
# 'reg_lambda': 25.581847755944548, 'subsample': 0.5}}

#2. 수정한 파라미터로 이용해서 재조정!!!!

# {'target': 1.0, 'params': {'colsample_bytree': 0.2625067497745618, 'max_depth': 12.50686297327293, 'min_child_weight': 0.16586485962120545, 'reg_alpha': 9.43520604988787,
# 'reg_lambda': 26.613082639937048, 'subsample': 1.880564370248726}}


model = XGBClassifier(n_estimators = 500, learning_rate= 0.02, colsample_bytree =max(min(0.2625067497745618,1),0) ,
max_depth=int(round(12.50686297327293)), min_child_weight =int(round(0.16586485962120545)),
reg_alpha= max(9.43520604988787,0), reg_lambda=max(26.613082639937048,0), subsample=max(min(1.880564370248726,1),0))

model.fit(x_train, y_train)
y_pred = model.predict(x_test)
score = accuracy_score(y_test, y_pred)
print('파마리터 수정 후 score : ', score)

# 파마리터 수정 후 score :  0.9166666666666666
