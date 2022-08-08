from re import S
import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
       train_size=0.9, shuffle=True, random_state=1234)

parameters = [
    {'RF__n_estimators' : [100,200,300], 'RF__max_depth' : [6,10]},                      
    {'RF__max_depth' : [6, 8], 'RF__min_samples_leaf' : [3, 5, 7]},         
    {'RF__min_samples_leaf' : [3, 5, 7], 'RF__min_samples_split' : [2, 3, 5]},  
    {'RF__min_samples_split' : [2, 3, 5]},                                     
    {'RF__n_jobs' : [-1, 2, 4],'RF__min_samples_split' : [2, 3, 5]}]

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
model = RandomizedSearchCV(pipe, parameters, cv=kfold, verbose=1)

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

# model.score :  0.9589342879763175
# accuracy_score :  0.9589342879763175



