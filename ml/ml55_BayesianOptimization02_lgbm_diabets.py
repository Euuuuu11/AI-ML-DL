from bayes_opt import BayesianOptimization
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np

from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import warnings
warnings.filterwarnings('ignore')

#1.데이터
datasets = load_diabetes()
x, y = datasets.data, datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=1234, train_size=0.8
)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
bayesian_params = {
    'max_depth' : (3, 9),
    'num_leaves' : (24, 64),
    'min_child_samples' : (51, 57),
    'min_child_weight' : (47, 53),
    'subsample' : (0.4, 1.3),
    'colsample_bytree' : (0.2, 0.8),
    'max_bin' : (469, 475),
    'num_leaves' : (36, 42),
    'reg_lambda' : (7, 13),
    'reg_alpha' : (47, 53),
}
def lgb_hamsu(max_depth, num_leaves, min_child_samples, min_child_weight,
              subsample, colsample_bytree, max_bin, reg_lambda, reg_alpha):
    params = {
        'n_estimators':500, "learning_rate":0.02,
        'max_depth' : int(round(max_depth)),                    # 무조건 정수형
        'num_leaves' : int(round(num_leaves)),
        'min_child_samples' : int(round(min_child_samples)),
        'min_child_weight' : int(round(min_child_weight)),      
        'subsample' : max(min(subsample, 1), 0),                # 0~1 사이의 값이 들어오
        'colsample_bytree' : max(min(colsample_bytree, 1), 0),  
        'max_bin' : max(int(round(max_bin)),10),                # 무조건 10이상 정수형
        'reg_lambda' : int(round(reg_lambda,0)),                # 무조건 양수만
        'reg_alpha' : int(round(reg_alpha,0))
        
    }
    
    # *여러개의인자를받겠다
    # **키워드받겠다(딕셔너리형태)
    model = LGBMRegressor(**params)
    
    model.fit(x_train, y_train,
              eval_set=[(x_train, y_train), (x_test, y_test)],
              eval_metric='rmse',
              verbose=0,
              early_stopping_rounds=50
              )
    
    y_predict = model.predict(x_test)
    results = r2_score(y_test, y_predict)
    
    return results

lgb_bo = BayesianOptimization(f=lgb_hamsu,
                              pbounds= bayesian_params,
                              random_state=123
                              )
lgb_bo.maximize(init_points=5, n_iter=100)

print(lgb_bo.max)

########################### [실습] ###########################
#1. 수정한 파라미터로 모델 만들어서 비교!!!!
# {'target': 0.49619581552622305, 'params': {'colsample_bytree': 0.5, 'max_bin': 472.3699843682731, 'max_depth': 6.0, 'min_child_samples': 54.48362394246377, 
# 'min_child_weight': 50.0, 'num_leaves': 39.33937472362355, 'reg_alpha': 50.0, 'reg_lambda': 10.0, 'subsample': 0.7590941633272226}}

#2. 수정한 파라미터로 이용해서 재조정!!!!
# {'target': 0.49905992730693316, 'params': {'colsample_bytree': 0.5189308243051031, 'max_bin': 472.1909655225812, 'max_depth': 6.806405751307926, 
# 'min_child_samples': 56.09659076446674, 'min_child_weight': 51.34673194916381, 'num_leaves': 39.6661410640655, 'reg_alpha': 51.33466029542133, 
# 'reg_lambda': 8.937753483119069, 'subsample': 0.7256097900600827}}
