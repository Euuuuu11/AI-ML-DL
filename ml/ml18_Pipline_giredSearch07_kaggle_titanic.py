from re import S
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
from tqdm import tqdm_notebook
from sklearn.preprocessing import LabelEncoder

#1. 데이터
path = './_data/kaggle_titanic/'
train_set =  pd.read_csv(path + 'train.csv',
                        index_col=0)
test_set = pd.read_csv(path + 'test.csv',   
                       index_col=0)

train_set = train_set.fillna(train_set.median())

drop_cols = ['Cabin']
train_set.drop(drop_cols, axis = 1, inplace =True)
test_set = test_set.fillna(test_set.mean())
train_set['Embarked'].fillna('S')
train_set = train_set.fillna(train_set.mean())

test_set.drop(drop_cols, axis = 1, inplace =True)
cols = ['Name','Sex','Ticket','Embarked']
for col in tqdm_notebook(cols):
    le = LabelEncoder()
    train_set[col]=le.fit_transform(train_set[col])
    test_set[col]=le.fit_transform(test_set[col])
    
x = train_set.drop(['Survived'],axis=1) 
y = train_set['Survived']

gender_submission = pd.read_csv(path + 'gender_submission.csv', #예측에서 쓰일 예정
                       index_col=0)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
       train_size=0.9, shuffle=True, random_state=1234)

parameters = [
    {'RF__n_estimators' : [100,200,300,400,500], 'RF__max_depth' : [6,10,12,14,16]},                      
    {'RF__max_depth' : [6, 8, 10, 12, 14], 'RF__min_samples_leaf' : [3, 5, 7, 10, 12]},         
    {'RF__min_samples_leaf' : [3, 5, 7, 10, 12], 'RF__min_samples_split' : [2, 3, 5, 10, 12]},  
    {'RF__min_samples_split' : [2, 3, 5, 10, 12]},                                     
    {'RF__n_jobs' : [-1, 2, 4, 6],'RF__min_samples_split' : [2, 3, 5, 10, 12]}]

from sklearn.model_selection import KFold, StratifiedKFold
n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=66)

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

#2. 모델
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline, Pipeline

# model = RandomForestClassifier()
# model = make_pipeline(MinMaxScaler(), RandomForestClassifier())
pipe = Pipeline([('minmax', MinMaxScaler()),('RF', RandomForestClassifier())],
                verbose=1)

#3. 훈련
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV

# model = GridSearchCV(RandomForestClassifier, parameters, cv=5, verbose=1)
model = HalvingRandomSearchCV(pipe, parameters, cv=kfold, verbose=1)

model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)
print('model.score : ', result)

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('accuracy_score : ', acc)

# print(x_test)
# print(y_predict)


# model.score :  0.8777777777777778
# accuracy_score :  0.8777777777777778

