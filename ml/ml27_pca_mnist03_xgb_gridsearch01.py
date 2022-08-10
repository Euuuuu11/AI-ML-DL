import numpy as np
from sklearn.decomposition import PCA
from keras.datasets import mnist 
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold , StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBClassifier,XGBRFRegressor        # activate tf282gpu > pip install xgboost
from sklearn.experimental import enable_halving_search_cv   # 실험버전사용할때 사용. 
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, HalvingRandomSearchCV
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV

(x_train,y_train),(x_test,y_test) =mnist.load_data()       # _를 사용하면 사용하지않겠다. 
# print(x_train.shape,x_test.shape)   # (60000, 28, 28) (10000, 28, 28)
 
x_train = x_train.reshape(60000,784)
x_test = x_test.reshape(10000,784)

pca = PCA(n_components=331)   
x_train= pca.fit_transform(x_train)
x_test= pca.transform(x_test) 
 
# print(x_train.shape,x_test.shape)       # (60000, 28, 28) (10000, 28, 28)

n_splits =5 
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

parameters = [ 
    {"n_estimators":[100,200,300],"learning_rate":[0.1,0.3,0.001,0.01],
     "max_depth":[4,5,6]},
    {"n_estimators":[90,100,110],"learning_rate":[0.1,0.001,0.01],
     "max_depth":[4,5,6],"colsample_bytree":[0.1,0.001,0.5]},
    {"n_estimators":[90,110],"learning_rate":[0.1,0.001,0.5],
     "max_depth":[4,5,6],"colsample_bytree":[0.6,0.9,1],
     "colsample_bylevel":[0.6,0.7,0.9] }
]

#2. 모델구성 
model =GridSearchCV(XGBClassifier(tree_method='gpu_hist'),
                    parameters, cv=kfold,verbose=1,      
                    refit=True,n_jobs=-1)

#3. 훈련
import time
start = time.time()
model.fit(x_train,y_train)
end = time.time()

#4. 예측
result = model.score(x_test,y_test)
print(model,) 
print("결과:",result)
print("걸린시간",end-start)