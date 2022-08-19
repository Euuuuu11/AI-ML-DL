from cgi import test
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

#1. 데이터
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

print(train_set)# [10886 rows x 13 columns]
print(test_set)# [6493 rows x 12 columns]

##########################################


x = train_set.drop(['count'], axis=1)  # drop 데이터에서 ''사이 값 빼기
print(x)
print(x.columns)
print(x.shape) # (10886, 12)
y = train_set['count'] 

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=1234,
)

kfold = KFold(n_splits=5, shuffle=True, random_state=1234)

#2. 모델
model = make_pipeline(StandardScaler(),
                      RandomForestRegressor()
                      )
model.fit(x_train,y_train)

print('just score : ',model.score(x_test,y_test))
# 0.7665382927362877

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, x_train, y_train, cv=kfold, scoring='r2')
print('just cv : ', scores)
print('just cv n/1 : ', np.mean(scores))

########################################## PolynomialFeatures 후 ##########################################
from sklearn.preprocessing import PolynomialFeatures

pf = PolynomialFeatures(degree=2, include_bias=False)
xp = pf.fit_transform(x)
# print(xp.shape)     # (506, 105)

x_train, x_test, y_train, y_test = train_test_split(
    xp, y, test_size=0.2, random_state=1234,
)
model = make_pipeline(StandardScaler(),
                      RandomForestRegressor()
                      )
model.fit(x_train,y_train)

print('poly score : ',model.score(x_test,y_test))

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, x_train, y_train, cv=kfold, scoring='r2')
print('poly cv : ', scores)
print('poly cv n/1 : ', np.mean(scores))


# just score :  0.954297383179385
# just cv :  [0.94947054 0.95510241 0.95108891 0.93952763 0.93578759]
# just cv n/1 :  0.9461954174888041
# poly score :  0.9484993232476694
# poly cv :  [0.94769305 0.95438874 0.95031926 0.9378687  0.92859819]
# poly cv n/1 :  0.9437735871412727