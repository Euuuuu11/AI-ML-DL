# 캐글 바이크
from tabnanny import verbose
from typing import Counter
import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential,  load_model
from tensorflow.python.keras.layers import Activation, Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout,LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score, r2_score
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold, cross_val_score

#1. 데이터
path = './_data/kaggle_house/'
train_set = pd.read_csv(path + 'train.csv')
test_set = pd.read_csv(path + 'test.csv')
# print(train_set.shape) # (1460, 81)
# print(test_set.shape)  # (1459, 80)


# 수치형 변수와 범주형 변수 찾기
numerical_feats = train_set.dtypes[train_set.dtypes != "object"].index
categorical_feats = train_set.dtypes[train_set.dtypes == "object"].index
# print("Number of Numberical features: ", len(numerical_feats)) # 38
# print("Number of Categorical features: ", len(categorical_feats)) # 43

# 변수명 출력
print(train_set[numerical_feats].columns)      
print("*"*80)
print(train_set[categorical_feats].columns)

# Index(['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond',
#        'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
#        'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
#        'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
#        'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
#        'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
#        'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
#        'MoSold', 'YrSold', 'SalePrice'],
#       dtype='object')
# *******************************************************************************
# Index(['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
#        'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
#        'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
#        'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
#        'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
#        'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
#        'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
#        'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
#        'SaleType', 'SaleCondition'],
#       dtype='object')


# 이상치 탐색 및 제거
def detect_outliers(df, n, features):
    outlier_indices = []
    for col in features:
        Q1 = np.percentile(df[col], 25)
        Q3 = np.percentile(df[col], 75)
        IQR = Q3 - Q1
        
        outlier_step = 1.5 * IQR
        
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_col)
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)
        
    return multiple_outliers
        
Outliers_to_drop = detect_outliers(train_set, 2,['Id', 'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual',
       'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
       'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF',
       'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
       'MiscVal', 'MoSold', 'YrSold'])

# categorical_feats 살펴보기
# for catg in list(categorical_feats) :
#     print(train_set[catg].value_counts())
#     print('#'*50)

# saleprice와 관련이 큰 변수들 끼리 정리.
num_strong_corr = ['SalePrice','OverallQual','TotalBsmtSF','GrLivArea','GarageCars',
                   'FullBath','YearBuilt','YearRemodAdd']

num_weak_corr = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallCond', 'MasVnrArea', 'BsmtFinSF1',
                 'BsmtFinSF2', 'BsmtUnfSF', '1stFlrSF', '2ndFlrSF','LowQualFinSF', 'BsmtFullBath',
                 'BsmtHalfBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
                 'Fireplaces', 'GarageYrBlt', 'GarageArea', 'WoodDeckSF','OpenPorchSF',
                 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']

catg_strong_corr = ['MSZoning', 'Neighborhood', 'Condition2', 'MasVnrType', 'ExterQual',
                    'BsmtQual','CentralAir', 'Electrical', 'KitchenQual', 'SaleType']

catg_weak_corr = ['Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 
                  'LandSlope', 'Condition1',  'BldgType', 'HouseStyle', 'RoofStyle', 
                  'RoofMatl', 'Exterior1st', 'Exterior2nd', 'ExterCond', 'Foundation', 
                  'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 
                  'HeatingQC', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 
                  'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 
                  'SaleCondition' ]


# 데이터 결측 처리
cols_fillna = ['PoolQC','MiscFeature','Alley','Fence','MasVnrType','FireplaceQu',
               'GarageQual','GarageCond','GarageFinish','GarageType', 'Electrical',
               'KitchenQual', 'SaleType', 'Functional', 'Exterior2nd', 'Exterior1st',
               'BsmtExposure','BsmtCond','BsmtQual','BsmtFinType1','BsmtFinType2',
               'MSZoning', 'Utilities']
# NaN을 없다는 의미의 None 데이터로 바꾼다.
for col in cols_fillna:
    train_set[col].fillna('None',inplace=True)
    test_set[col].fillna('None',inplace=True)

# 결측치의 처리정도 확인.
total = train_set.isnull().sum().sort_values(ascending=False)
percent = (train_set.isnull().sum()/train_set.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(5)

# 결측치 확인
# print(train_set.isnull().sum().sum(), test_set.isnull().sum().sum())

# 수치형 변수들 평균값으로 대체.
train_set.fillna(train_set.mean(), inplace=True)
test_set.fillna(test_set.mean(), inplace=True)

total = train_set.isnull().sum().sort_values(ascending=False)
percent = (train_set.isnull().sum()/train_set.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(5)

# 결측치 확인
# print(train_set.isnull().sum().sum(), test_set.isnull().sum().sum())

# SalePrice'와의 상관관계가 약한 모든 변수를 삭제
id_test = test_set['Id']

to_drop_num  = num_weak_corr
to_drop_catg = catg_weak_corr

cols_to_drop = ['Id'] + to_drop_num + to_drop_catg 

for df in [train_set, test_set]:
    df.drop(cols_to_drop, inplace= True, axis = 1)
    
# print(train_set.head())

# 수치형 변환을 위해 각 변수들의 범주들을 그룹화 시킴.

# 'MSZoning'
msz_catg2 = ['RM', 'RH']
msz_catg3 = ['RL', 'FV'] 


# Neighborhood
nbhd_catg2 = ['Blmngtn', 'ClearCr', 'CollgCr', 'Crawfor', 'Gilbert', 'NWAmes', 'Somerst', 'Timber', 'Veenker']
nbhd_catg3 = ['NoRidge', 'NridgHt', 'StoneBr']

# Condition2
cond2_catg2 = ['Norm', 'RRAe']
cond2_catg3 = ['PosA', 'PosN'] 

# SaleType
SlTy_catg1 = ['Oth']
SlTy_catg3 = ['CWD']
SlTy_catg4 = ['New', 'Con']

# 범주별 수치형 변환 실행
for df in [train_set, test_set]:
    
    df['MSZ_num'] = 1  
    df.loc[(df['MSZoning'].isin(msz_catg2) ), 'MSZ_num'] = 2    
    df.loc[(df['MSZoning'].isin(msz_catg3) ), 'MSZ_num'] = 3        
    
    df['NbHd_num'] = 1       
    df.loc[(df['Neighborhood'].isin(nbhd_catg2) ), 'NbHd_num'] = 2    
    df.loc[(df['Neighborhood'].isin(nbhd_catg3) ), 'NbHd_num'] = 3    

    df['Cond2_num'] = 1       
    df.loc[(df['Condition2'].isin(cond2_catg2) ), 'Cond2_num'] = 2    
    df.loc[(df['Condition2'].isin(cond2_catg3) ), 'Cond2_num'] = 3    
    
    df['Mas_num'] = 1       
    df.loc[(df['MasVnrType'] == 'Stone' ), 'Mas_num'] = 2 
    
    df['ExtQ_num'] = 1       
    df.loc[(df['ExterQual'] == 'TA' ), 'ExtQ_num'] = 2     
    df.loc[(df['ExterQual'] == 'Gd' ), 'ExtQ_num'] = 3     
    df.loc[(df['ExterQual'] == 'Ex' ), 'ExtQ_num'] = 4     
   
    df['BsQ_num'] = 1          
    df.loc[(df['BsmtQual'] == 'Gd' ), 'BsQ_num'] = 2     
    df.loc[(df['BsmtQual'] == 'Ex' ), 'BsQ_num'] = 3     
 
    df['CA_num'] = 0          
    df.loc[(df['CentralAir'] == 'Y' ), 'CA_num'] = 1    

    df['Elc_num'] = 1       
    df.loc[(df['Electrical'] == 'SBrkr' ), 'Elc_num'] = 2 


    df['KiQ_num'] = 1       
    df.loc[(df['KitchenQual'] == 'TA' ), 'KiQ_num'] = 2     
    df.loc[(df['KitchenQual'] == 'Gd' ), 'KiQ_num'] = 3     
    df.loc[(df['KitchenQual'] == 'Ex' ), 'KiQ_num'] = 4      
    
    df['SlTy_num'] = 2       
    df.loc[(df['SaleType'].isin(SlTy_catg1) ), 'SlTy_num'] = 1  
    df.loc[(df['SaleType'].isin(SlTy_catg3) ), 'SlTy_num'] = 3  
    df.loc[(df['SaleType'].isin(SlTy_catg4) ), 'SlTy_num'] = 4
    
# 기존 범주형 변수와 새로 만들어진 수치형 변수 역시 유의하지 않은 것들 삭제.
train_set.drop(['MSZoning','Neighborhood' , 'Condition2', 'MasVnrType', 'ExterQual', 'BsmtQual','CentralAir', 'Electrical', 'KitchenQual', 'SaleType', 'Cond2_num', 'Mas_num', 'CA_num', 'Elc_num', 'SlTy_num'], axis = 1, inplace = True)
test_set.drop(['MSZoning', 'Neighborhood' , 'Condition2', 'MasVnrType', 'ExterQual', 'BsmtQual','CentralAir', 'Electrical', 'KitchenQual', 'SaleType', 'Cond2_num', 'Mas_num', 'CA_num', 'Elc_num', 'SlTy_num'], axis = 1, inplace = True)
  
x = train_set.drop(['SalePrice'], axis=1)
y = train_set['SalePrice']

# print(x.shape)  # (1460, 12)
# print(y.shape)  # (1460,  )

x_train, x_test, y_train, y_test = train_test_split(x,y,
             train_size=0.9, shuffle=True, random_state=777)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

scaler = MinMaxScaler()

scaler.fit(x_train)
print(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# print(x.shape,y.shape) # (1460, 12) (1460,)
# print(x_train.shape,x_test.shape) # (1314, 12) (146, 12)



#2. 모델구성
# allAlogrithms = all_estimators(type_filter='classifier')
allAlogrithms = all_estimators(type_filter='regressor')

# [('AdaBoostClassifier', <class 'sklearn.ensemble._weight_boosting.AdaBoostClassifier'>

print('allAlogrithms : ', allAlogrithms)    # 딕셔너리들이 list 형태로 묶여져있다.
print('모델의 개수 : ', len(allAlogrithms))  # 모델의 개수 :  41

# [예외처리] 에러가 떳을 때 무시하고, 넘어가겠다. 
for (name, algorithm) in allAlogrithms:
    try :
        model = algorithm()
        model.fit(x_train, y_train)
        
        scores = cross_val_score(model, x_test, y_test, cv=5)  
        print(name , scores, '\n cross_val_score : ', round(np.mean(scores), 4))
    except :
        # continue    
        print(name, '은 안나온 놈!!')    

# 모델의 개수 :  54
# ARDRegression [-0.00052472 -0.00174569 -0.29626664 -0.06950134 -0.04377093] 
#  cross_val_score :  -0.0824
# AdaBoostRegressor [0.67752154 0.74844881 0.79447019 0.70525348 0.82315842] 
#  cross_val_score :  0.7498
# BaggingRegressor [0.67130615 0.71016201 0.75925381 0.75366259 0.86829193] 
#  cross_val_score :  0.7525
# BayesianRidge [-5.24711928e-04  7.23912069e-01 -2.96266629e-01  7.53259406e-01
#  -4.37709258e-02]
#  cross_val_score :  0.2273
# CCA [-0.63972528  0.42842067 -0.51468559  0.41304422 -0.05151049] 
#  cross_val_score :  -0.0729
# DecisionTreeRegressor [0.38013684 0.53577134 0.72649425 0.53176183 0.64569117] 
#  cross_val_score :  0.564
# DummyRegressor [-0.00052474 -0.00174571 -0.29626666 -0.06950135 -0.04377095]
#  cross_val_score :  -0.0824
# ElasticNet [0.39889435 0.27348995 0.23714374 0.28298281 0.41007767] 
#  cross_val_score :  0.3205
# ElasticNetCV [ 0.03456155  0.02408353 -0.25362109 -0.04217638 -0.00251994] 
#  cross_val_score :  -0.0479
# ExtraTreeRegressor [0.45811385 0.6286691  0.39095888 0.40710277 0.77490613]
#  cross_val_score :  0.532
# ExtraTreesRegressor [0.72972277 0.64457491 0.81393704 0.70692709 0.86414296] 
#  cross_val_score :  0.7519
# GammaRegressor [0.27425161 0.20181295 0.05632234 0.15900371 0.24453666] 
#  cross_val_score :  0.1872
# GaussianProcessRegressor [-284.24080806 -160.71998114 -401.63565712 -269.74043217 -242.78304828] 
#  cross_val_score :  -271.824
# GradientBoostingRegressor [0.72261107 0.7547233  0.76576623 0.74635153 0.84196168] 
#  cross_val_score :  0.7663
# HistGradientBoostingRegressor [0.65743551 0.63960769 0.76378713 0.64939276 0.85522542] 
#  cross_val_score :  0.7131
# HuberRegressor [0.64052044 0.62975267 0.74969197 0.61857355 0.84126112] 
#  cross_val_score :  0.696
# IsotonicRegression 은 안나온 놈!!
# KNeighborsRegressor [0.52543426 0.53341202 0.70243433 0.63637712 0.78166606] 
#  cross_val_score :  0.6359
# KernelRidge [0.71432384 0.68615158 0.72135003 0.69867522 0.89546905] 
#  cross_val_score :  0.7432
# Lars [0.78227998 0.63282201 0.7440322  0.76755903 0.90300181] 
#  cross_val_score :  0.7659
# LarsCV [0.78354028 0.68297516 0.76307307 0.76349945 0.91318362] 
#  cross_val_score :  0.7813
# Lasso [0.78227301 0.71792224 0.74418118 0.76748242 0.90308541]
#  cross_val_score :  0.783
# LassoCV [0.76655198 0.72156449 0.76415248 0.75694413 0.91159662] 
#  cross_val_score :  0.7842
# LassoLars [0.78227322 0.71794157 0.7445371  0.76751405 0.90312347]
#  cross_val_score :  0.7831
# LassoLarsCV [0.78354028 0.72021847 0.78381233 0.76349945 0.91318362] 
#  cross_val_score :  0.7929
# LassoLarsIC [0.75823317 0.71079527 0.77950582 0.74870988 0.91179557]
#  cross_val_score :  0.7818
# LinearRegression [0.78227998 0.71780958 0.7440322  0.76755903 0.90300181] 
#  cross_val_score :  0.7829
# LinearSVR [-13.09448683  -5.24979778 -11.26592089 -10.42298826 -10.6418173 ]
#  cross_val_score :  -10.135
# MLPRegressor [-13.16440609  -5.27961142 -11.33586164 -10.4783135  -10.70032573] 
#  cross_val_score :  -10.1917
# MultiOutputRegressor 은 안나온 놈!!
# MultiTaskElasticNet 은 안나온 놈!!
# MultiTaskElasticNetCV 은 안나온 놈!!
# MultiTaskLasso 은 안나온 놈!!
# MultiTaskLassoCV 은 안나온 놈!!
# NuSVR [-0.00449514 -0.00100234 -0.14695238 -0.12476692 -0.09812618] 
#  cross_val_score :  -0.0751
# OrthogonalMatchingPursuit [0.49914641 0.59937387 0.46815955 0.58010972 0.8266312 ]
#  cross_val_score :  0.5947
# OrthogonalMatchingPursuitCV [0.74966417 0.68671289 0.73444288 0.7568224  0.91002468] 
#  cross_val_score :  0.7675
# PLSCanonical [-3.64994517 -0.58768975 -2.78175554 -1.12228169 -2.2077136 ]
#  cross_val_score :  -2.0699
# PLSRegression [0.78724063 0.73908323 0.76042279 0.78160239 0.91052918] 
#  cross_val_score :  0.7958
# PassiveAggressiveRegressor [0.51084134 0.47945764 0.70068952 0.49538414 0.68118433] 
#  cross_val_score :  0.5735
# PoissonRegressor [0.78980645 0.78641579 0.77791718 0.77231993 0.90288836] 
#  cross_val_score :  0.8059
# RANSACRegressor [0.69420087 0.73435671 0.46173347 0.80656283 0.92276127] 
#  cross_val_score :  0.7239
# RadiusNeighborsRegressor [0.47596403 0.40624358 0.53616497 0.44048509 0.62425402] 
#  cross_val_score :  0.4966
# RandomForestRegressor [0.75245991 0.70994757 0.82267433 0.75305953 0.86640621] 
#  cross_val_score :  0.7809
# RegressorChain 은 안나온 놈!!
# Ridge [0.71433395 0.68626994 0.72054268 0.68524356 0.89321181]
#  cross_val_score :  0.7399
# RidgeCV [0.77222638 0.72529913 0.73345945 0.75296519 0.9081626 ] 
#  cross_val_score :  0.7784
# SGDRegressor [0.70698171 0.69714987 0.68012375 0.68816158 0.89405124] 
#  cross_val_score :  0.7333
# SVR [-0.15848874 -0.00203344 -0.09871538 -0.21651209 -0.19010689] 
#  cross_val_score :  -0.1332
# StackingRegressor 은 안나온 놈!!
# TheilSenRegressor [0.79486858 0.6967457  0.7957323  0.78221721 0.90883202] 
#  cross_val_score :  0.7957
# TransformedTargetRegressor [0.78227998 0.71780958 0.7440322  0.76755903 0.90300181]
#  cross_val_score :  0.7829
# TweedieRegressor [0.27195709 0.17476793 0.05233941 0.16219118 0.25119867] 
#  cross_val_score :  0.1825
# VotingRegressor 은 안나온 놈!!