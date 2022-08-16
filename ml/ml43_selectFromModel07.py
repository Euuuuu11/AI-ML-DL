#[실습] girdSearchfrom sklearn.datasets import load_breast_cancer

from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold,StratifiedKFold,train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
import time 
from sklearn.feature_selection import SelectFromModel   # 모델을 선택.
import numpy as np
import pandas as pd
#1.데이터
path = './_data/kaggle_titanic/'
train_set = pd.read_csv(path + 'train.csv',index_col =0)
test_set = pd.read_csv(path + 'test.csv', index_col=0)

##########전처리############
train_test_data = [train_set, test_set]
sex_mapping = {"male":0, "female":1}
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)

print(dataset)

for dataset in train_test_data:
    # 가족수 = 형제자매 + 부모님 + 자녀 + 본인
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    dataset['IsAlone'] = 1
    
    # 가족수 > 1이면 동승자 있음
    dataset.loc[dataset['FamilySize'] > 1, 'IsAlone'] = 0

for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
embarked_mapping = {'S':0, 'C':1, 'Q':2}
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)

for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract('([\w]+)\.', expand=False)
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].apply(lambda x: 0 if x=="Mr" else 1 if x=="Miss" else 2 if x=="Mrs" else 3 if x=="Master" else 4)

train_set['Cabin'] = train_set['Cabin'].str[:1]
for dataset in train_test_data:
    dataset['Age'].fillna(dataset.groupby("Title")["Age"].transform("median"), inplace=True)
for dataset in train_test_data:
    dataset['Agebin'] = pd.cut(dataset['Age'], 5, labels=[0,1,2,3,4])
for dataset in train_test_data:
    dataset["Fare"].fillna(dataset.groupby("Pclass")["Fare"].transform("median"), inplace=True)
for dataset in train_test_data:
    dataset['Farebin'] = pd.qcut(dataset['Fare'], 4, labels=[0,1,2,3])
    drop_column = ['Name', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin']

for dataset in train_test_data:
    dataset = dataset.drop(drop_column, axis=1, inplace=True)
print(train_set.head())


x = train_set.drop(['Survived'], axis=1,)
y = train_set['Survived']

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.8, random_state=123)

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
acc = accuracy_score(y_test,y_predict)
print("최종 acc :", acc)


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
    selection_model =XGBClassifier(n_jobs=-1,
                                   random_state=123,                 
                                   n_estimators=100,
                                   learning_rate=0.1,
                                   max_depth=3,
                                   gamma=1)
    
    selection_model.fit(select_x_train,y_train)
    
    y_predict = selection_model.predict(select_x_test)
    score = accuracy_score(y_test,y_predict)
    
    print('Thresh=%.3f, n=%d, acc: %.2f%%'
          %(thresh, select_x_train.shape[1],score*100))


# (712, 5) (179, 5)
# Thresh=0.044, n=5, acc: 86.03%