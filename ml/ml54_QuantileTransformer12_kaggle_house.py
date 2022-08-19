from cgi import test
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
from sklearn.preprocessing import LabelEncoder


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
    x, y, test_size=0.2, random_state=1234,
)
# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# scaler = MaxAbsScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# scaler = RobustScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# scaler = QuantileTransformer()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

scaler = PowerTransformer(method = 'yeo-johnson')    # 디폴트
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# scaler = PowerTransformer(method = 'box-cox')
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

#2. 모델
model = LinearRegression()
# model = RandomForestRegressor()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
y_predict = model.predict(x_test)
results = r2_score(y_test, y_predict)
print("그냥 결과 : ", round(results, 4))

####################################################################
# StandardScaler    그냥 결과 :  0.8534
# MinMaxScaler      그냥 결과 :  0.8533
# MaxAbsScaler      그냥 결과 :  0.8534
# RobustScaler      그냥 결과 :  0.8534
# QuantileTransformer     그냥 결과 :  0.8247
# PowerTransformer(method = 'yeo-johnson')   그냥 결과 :  0.8485

# ############################ 로그 변환 ############################ 
# df = pd.DataFrame(datasets.data, columns=[datasets.feature_names])
# print(df)

# # df.plot.box()
# # plt.title('boston')
# # plt.xlabel('Feature')
# # plt.ylabel('데이터값')
# # plt.show()

# # print(df['B'].head())                 #  그냥 결과 :  0.7665
# df['B'] = np.log1p(df['B'])           #  그냥 결과 :  0.7711
# # print(df['B'].head())

# # df['CRIM'] = np.log1p(df['CRIM'])   # 로그변환 결과 :  0.7596
# df['ZN'] = np.log1p(df['ZN'])       # 로그변환 결과 :  0.7734
# df['TAX'] = np.log1p(df['TAX'])     # 로그변환 결과 :  0.7669
#                                     # 3개 모두 쓰면 : 0.7785
                                    
# x_train, x_test, y_train, y_test = train_test_split(
#     df, y, test_size=0.2, random_state=1234,
# )
# # scaler = StandardScaler()
# # x_train = scaler.fit_transform(x_train)
# # x_test = scaler.transform(x_test)



# #2. 모델
# model = LinearRegression()
# # model = RandomForestRegressor()

# #3. 훈련
# model.fit(x_train, y_train)

# #4. 평가, 예측
# y_predict = model.predict(x_test)
# results = r2_score(y_test, y_predict)
# print("로그변환 결과 : ", round(results, 4))

# # LinearRegression
# # 그냥 결과 :  0.7711

# # RandomForestRegressor
# # 그냥 결과 :  0.9153