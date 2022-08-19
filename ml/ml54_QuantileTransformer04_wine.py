from cgi import test
from sklearn.datasets import load_wine
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

datasets = load_wine()
x,y = datasets.data, datasets.target
# print(x.shape, y.shape)     # (506, 13) (506,)

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
model = LogisticRegression()
# model = RandomForestClassifier()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
y_predict = model.predict(x_test)
results = r2_score(y_test, y_predict)
print("그냥 결과 : ", round(results, 4))

####################################################################
# StandardScaler    그냥 결과 :  0.9473
# MinMaxScaler      그냥 결과 :  0.9473
# MaxAbsScaler      그냥 결과 :  0.8419
# RobustScaler      그냥 결과 :  0.9473
# QuantileTransformer     그냥 결과 :  0.9473
# PowerTransformer(method = 'yeo-johnson')   그냥 결과 :  0.9473

# ############################ 로그 변환 ############################ 
# df = pd.DataFrame(datasets.data, columns=[datasets.feature_names])
# print(df)

# # df.plot.box()
# # plt.title('wine')
# # plt.xlabel('Feature')
# # plt.ylabel('데이터값')
# # plt.show()

# # print(df['B'].head())                 #  그냥 결과 :  0.7665
# df['magnesium'] = np.log1p(df['magnesium'])           #  그냥 결과 :  0.7711
# # print(df['B'].head())

# # df['CRIM'] = np.log1p(df['CRIM'])   # 로그변환 결과 :  0.7596
# # df['ZN'] = np.log1p(df['ZN'])       # 로그변환 결과 :  0.7734
# # df['TAX'] = np.log1p(df['TAX'])     # 로그변환 결과 :  0.7669
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

# # 그냥 결과 :  0.9473
# # 로그변환 결과 :  0.8094