#[실습] girdSearchfrom sklearn.datasets import load_breast_cancer

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold,StratifiedKFold,train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
import time 

#1. 데이터 
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

print(x.shape,y.shape)          # (569, 30) (569,)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.8, random_state=12, stratify=y)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5

kfold = StratifiedKFold(n_splits=n_splits ,shuffle=True, random_state=123)


#2. 모델 

model = XGBClassifier(random_state=123,                 #위에있는 파라미터를 모델안에 넣을때 하는 방법
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    gamma=1

)

# model = GridSearchCV(xgb, parameters, cv =kfold, n_jobs=8)

import time
start = time.time()


#3. 훈련
# model.fit(x_train,y_train, early_stopping_rounds =200,
#           eval_set = [(x_train,y_train),(x_test,y_test)],   # 훈련 + 학습 # 뒤에걸 인지한다
#           eval_metric='error',          
#           # rmse,mae,mrmsle...  회귀             
#           # errror, auc...      이진  
#           # merror,mlogloss..   다중
#           )



# end = time.time()
##############################불러오기 // 2.모델 3. 훈련 #############################

import pickle
import joblib

# 모델저장
path = 'D:\study_data\_save/_xg/'
# model = joblib.load((path+'m40_joblib1_save.dat'))   # wb 불러오기(save) rb 가져오기(load) 
model.load_model(path +'m41_xgb1_save_model.dat')

#4. 평가 예측
results= model.score(x_test,y_test)
print("결과 :",results)


from sklearn.metrics import accuracy_score,r2_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test,y_predict)
print("최종 acc :", acc)

# import pickle

# # 모델저장
# path = 'D:\study_data\_save/_xg/'
# pickle.dump(model, open(path+'m39_pickle1_save.dat','wb'))




