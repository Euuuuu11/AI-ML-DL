from bayes_opt import BayesianOptimization
from lightgbm import LGBMRegressor, LGBMClassifier
import numpy as np

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import warnings
warnings.filterwarnings('ignore')
from xgboost import XGBClassifier, XGBRegressor

#1.데이터
datasets = load_iris()
x, y = datasets.data, datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=1234, train_size=0.8
)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
bayesian_params = {      
    'max_depth':(1, 10),                                         
    'min_child_weight':(0, 200),             
    'subsample':(0.5 ,1),                        
    'colsample_bytree':(0.5,1),                                 
    'reg_alpha':(0.0001,100),                           
    'reg_lambda':(0.01,100)
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
#1. 수정한 파라미터로 모델 만들어서 비교!!!!
# {'target': 0.7429951690821256, 'params': {'colsample_bytree': 0.6312952464075539, 'max_bin': 11.779904623902896, 'max_depth': 12.90838185015963, 
# 'min_child_samples': 144.81187361329597, 'min_child_weight': 9.877363607133844, 'num_leaves': 30.481074123605598, 'reg_alpha': 2.377888753370187, 
# 'reg_lambda': 5.078293840286648, 'subsample': 0.593082760434454}}

#2. 수정한 파라미터로 이용해서 재조정!!!!

# 'target': 0.8164251207729468, 'params': {'colsample_bytree': 0.3, 'max_bin': 8.0, 'max_depth': 9.0, 'min_child_samples': 144.98158214197016, 
# 'min_child_weight': 10.246864012756383, 'num_leaves': 33.0, 'reg_alpha': 0.1, 'reg_lambda': 2.0, 'subsample': 1.0}}