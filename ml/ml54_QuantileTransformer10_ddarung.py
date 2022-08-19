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

path = './_data/ddarung/'
train_set = pd.read_csv(path + 'train.csv',                 # + 명령어는 문자를 앞문자와 더해줌
                        index_col=0)                        # index_col=n n번째 컬럼을 인덱스로 인식

test_set = pd.read_csv(path + 'test.csv',                    # 예측에서 쓸거임                
                       index_col=0)

train_set = train_set.fillna(train_set.mean())       # dropna() : train_set 에서 na, null 값 들어간 행 삭제
test_set = test_set.fillna(test_set.mean()) # test_set 에서 이빨빠진데 바로  ffill : 위에서 가져오기 test_set.mean : 평균값

x = train_set.drop(['count'], axis=1)                    # drop 데이터에서 ''사이 값 빼기

y = train_set['count'] 
x = np.array(x)

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
# StandardScaler    그냥 결과 :  0.6059
# MinMaxScaler      그냥 결과 :  0.6059
# MaxAbsScaler      그냥 결과 :  0.6059
# RobustScaler      그냥 결과 :  0.6059
# QuantileTransformer     그냥 결과 :  0.6041
# PowerTransformer(method = 'yeo-johnson')   그냥 결과 :  0.5971

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