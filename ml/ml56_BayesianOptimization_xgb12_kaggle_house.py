from bayes_opt import BayesianOptimization
from lightgbm import LGBMRegressor, LGBMClassifier
import numpy as np
from tqdm import tqdm_notebook
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import warnings
warnings.filterwarnings('ignore')
from xgboost import XGBClassifier, XGBRegressor

#1.데이터
path = 'C:\study\_data\kaggle_house/' # ".은 현재 폴더"
train_set = pd.read_csv(path + 'train.csv',
                        index_col=0)
test_set = pd.read_csv(path + 'test.csv', #예측에서 쓸거야!!
                       index_col=0)
drop_cols = ['Alley', 'PoolQC', 'Fence', 'MiscFeature']
test_set.drop(drop_cols, axis = 1, inplace =True)
submission = pd.read_csv(path + 'sample_submission.csv',#예측에서 쓸거야!!
                       index_col=0)
# print(train_set)

# print(train_set.shape) #(1459, 10)

train_set.drop(drop_cols, axis = 1, inplace =True)
cols = ['MSZoning', 'Street','LandContour','Neighborhood','Condition1','Condition2',
                'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Foundation',
                'Heating','GarageType','SaleType','SaleCondition','ExterQual','ExterCond','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
                'BsmtFinType2','HeatingQC','CentralAir','Electrical','KitchenQual','Functional',
                'FireplaceQu','GarageFinish','GarageQual','GarageCond','PavedDrive','LotShape',
                'Utilities','LandSlope','BldgType','HouseStyle','LotConfig']

for col in tqdm_notebook(cols):
    le = LabelEncoder()
    train_set[col]=le.fit_transform(train_set[col])
    test_set[col]=le.fit_transform(test_set[col])


###### 결측치 처리 1.제거##### dropna 사용
# print(train_set.isnull().sum()) #각 컬럼당 결측치의 합계
train_set = train_set.fillna(train_set.median())
# print(train_set.isnull().sum())
# print(train_set.shape)
test_set = test_set.fillna(test_set.median())

x = train_set.drop(['SalePrice'],axis=1) #axis는 컬럼 
# print(x.columns)
# print(x.shape) #(1460, 75)

y = train_set['SalePrice']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=1234, train_size=0.8
)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
bayesian_params = {      
   'max_depth':(1,10),
    'min_child_weight' : (0,200),
    'subsample':(0.5,1),
    'colsample_bytree':(0.5,1),
    'reg_lambda' : (0.0001,100),
    'reg_alpha':(0.01,100)
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
    model = XGBRegressor(**params)
    
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
#1. 수정한 파라미터로 모델 만들어서 비교!!!

# {'target': 0.8803092160885161, 'params': {'colsample_bytree': 1.0, 'max_depth': 9.172342906790268, 'min_child_weight': 4.70870620090763,
# 'reg_alpha': 25.88191630163494, 'reg_lambda': 40.63970824337975, 'subsample': 1.0}}

#2. 수정한 파라미터로 이용해서 재조정!!!!
# {'target': 0.8823785605406569, 'params': {'colsample_bytree': 1.2636226668047854, 'max_depth': 11.465364467877958, 'min_child_weight': 3.1338154591221943,
# 'reg_alpha': 26.728172045587378, 'reg_lambda': 39.78985076919175, 'subsample': 1.2661537011692694}}



# model = XGBRegressor(n_estimators = 500, learning_rate= 0.02, colsample_bytree =max(min(1.2636226668047854,1),0) ,
# max_depth=int(round(11.465364467877958)), min_child_weight =int(round(3.1338154591221943)),
# reg_alpha= max(26.728172045587378,0), reg_lambda=max(39.78985076919175,0), subsample=max(min(1.2661537011692694,1),0))

# model.fit(x_train, y_train)
# y_pred = model.predict(x_test)
# score = r2_score(y_test, y_pred)
# print('파마리터 수정 후 score : ', score)

# 파마리터 수정 후 score :  0.8827718726390009
