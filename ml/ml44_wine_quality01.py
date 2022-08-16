import pandas as pd
from sklearn.model_selection import KFold,StratifiedKFold,train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

file = 'D:/study_data/_data/'

data = pd.read_csv(file + 'winequality-white.csv', index_col=None,
                   header=0, sep=';')  

# print(data.shape)   # (4898, 12)

x = data.values[:,0:11]
y = data.values[:,11]

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
# print(x.shape,y.shape)
# exit()
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.8, 
                                                    random_state=123,stratify=y)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5

kfold = StratifiedKFold(n_splits=n_splits ,shuffle=True, random_state=123)


model = RandomForestClassifier()
    # random_state=123,                 #위에있는 파라미터를 모델안에 넣을때 하는 방법
    #                 n_estimators=100,
    #                 learning_rate=0.1,
    #                 max_depth=3,
    #                 gamma=1)

import time
start = time.time()

model.fit(x_train,y_train)
        #   , early_stopping_rounds =200,
        #   eval_set = [(x_train,y_train),(x_test,y_test)],   # 훈련 + 학습 # 뒤에걸 인지한다
        #   eval_metric='merror')  

end = time.time()

#4. 평가 예측

results= model.score(x_test,y_test)
print("결과 :",results)
print("시간 :", end-start )

from sklearn.metrics import accuracy_score,r2_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test,y_predict)
print("최종 acc :", acc)

# 결과 : 0.7326530612244898
# 시간 : 0.5253987312316895
# 최종 acc : 0.7326530612244898