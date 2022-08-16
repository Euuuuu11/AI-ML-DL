#[실습] girdSearchfrom sklearn.datasets import load_breast_cancer

from sklearn.datasets import load_breast_cancer, load_diabetes,load_boston ,fetch_california_housing
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold,StratifiedKFold,train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
import time 
from sklearn.feature_selection import SelectFromModel   # 모델을 선택.
import pandas as pd
import numpy as np
#1. 데이터
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
x = np.delete(x,[2,3,4], axis=1)  

# x = np.delete(x,1, axis=1) 
# x = np.delete(x,4, axis=1) 

# y = np.delete(y,1, axis=1) 


# print(x.shape,y.shape)
# print(datasets.feature_names)


from sklearn.model_selection import train_test_split


x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.8, random_state=123)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5

kfold = StratifiedKFold(n_splits=n_splits ,shuffle=True, random_state=123)


#2. 모델 

model = XGBRegressor(random_state=123,                 #위에있는 파라미터를 모델안에 넣을때 하는 방법
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    gamma=1

)

# model = GridSearchCV(xgb, parameters, cv =kfold, n_jobs=8)

import time
start = time.time()



model.fit(x_train,y_train, early_stopping_rounds =200,
          eval_set = [(x_train,y_train),(x_test,y_test)],   # 훈련 + 학습 # 뒤에걸 인지한다
          eval_metric='error',          
          # rmse,mae,mrmsle...  회귀             
          # errror, auc...      이진  
          # merror,mlogloss..   다중
          )

end = time.time()

#4. 평가 예측

results= model.score(x_test,y_test)
print("결과 :",results)
print("시간 :", end-start )

from sklearn.metrics import accuracy_score,r2_score
y_predict = model.predict(x_test)
acc = r2_score(y_test,y_predict)
print("최종 r2_score :", acc)


print(model.feature_importances_)

# 결과 : -2.8392073911166262
# 시간 : 0.08741998672485352
# 최종 r2_score : -2.8392073911166262
# [0.03986917 0.04455113 0.25548902 0.07593288 0.04910125 0.04870857      
#  0.06075545 0.05339111 0.30488744 0.06731401]

thresholds =model.feature_importances_
print('=========================================')
for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    
    print(select_x_train.shape,select_x_test.shape)
#     =========================================
# (353, 10) (89, 10)        # 각 번째수 마다 작거나 큰 값을 반환 
# (353, 9) (89, 9)          # 피처에 중요도에 따라 10번을 돌려서 성능평가가 가능함
# (353, 2) (89, 2)
# (353, 3) (89, 3)
# (353, 7) (89, 7)
# (353, 8) (89, 8)
# (353, 5) (89, 5)
# (353, 6) (89, 6)
# (353, 1) (89, 1)
# (353, 4) (89, 4)
    selection_model = XGBRegressor(n_jobs=-1,
                                   random_state=123,                 
                                   n_estimators=100,
                                   learning_rate=0.1,
                                   max_depth=3,
                                   gamma=1)
    
    selection_model.fit(select_x_train,y_train)
    
    y_predict = selection_model.predict(select_x_test)
    score = r2_score(y_test,y_predict)
    
    print('Thresh=%.3f, n=%d, R2: %.2f%%'
          %(thresh, select_x_train.shape[1],score*100))

# (1167, 6) (292, 6)
# Thresh=0.043, n=6, R2: 75.76%