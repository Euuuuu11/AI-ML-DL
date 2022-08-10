import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, fetch_covtype, load_digits
from sklearn.decomposition import PCA 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xg
from tqdm import tqdm_notebook

# print('xgboost버전 : ', xg.__version__) # xgboost버전 :  1.6.1

'''
01. iris
02. cancer
03. diabets \\ 회귀
04. wine 
05. fetch_covtype
06. digits  
07. kaggle_titinic
'''

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
# print(x.shape, y.shape)

# x = x.drop(x.columns[[3]], axis=1)
# x = np.delete(x, 2, axis=1)
x = x.drop(x.columns[[2]], axis=1)
# datasets = load_breast_cancer()


x = datasets.data
y = datasets.target
# print(x.shape)              # (581012, 54)

# le = LabelEncoder()
# y = le.fit_transform(y)

# pca = PCA(n_components=20)       #   54 >10
# x = pca.fit_transform(x)

# lda = LinearDiscriminantAnalysis()
lda = LinearDiscriminantAnalysis()
lda.fit(x,y)
x = lda.transform(x)
print(x)

# pca_EVR = pca.explained_variance_ratio_
# cumsum = np.cumsum(pca_EVR)             
# print(cumsum)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,shuffle=True,random_state=123,
                                                    stratify=y)

# lda = LinearDiscriminantAnalysis()
# lda.fit(x_train,y_train)
# x_train = lda.transform(x_train)
# x_test = lda.transform(x_test)

# print(np.unique(y_train, return_counts=True))               # array([1, 2, 3, 4, 5, 6, 7] > 
                                                            # array([0, 1, 2, 3, 4, 5, 6]
                                                            
#2. 모델
from xgboost import XGBClassifier ,XGBRFRegressor
model = XGBRFRegressor(tree_method='gpu_hist',
                      predictor='gpu_predictor',
                      gpu_id=0)

#3. 훈련
import time
start = time.time()
model.fit(x_train,y_train)
end = time.time()

#4. 평가 예측

results= model.score(x_test,y_test)
print("결과 :",results)
print("시간 :", end-start )


# XGBClassifier
# 결과: 0.8695988915948814
# 시간 : 6.970503330230713

# XGBClassifier
# pca = PCA(n_components=10)
# 결과: 0.8406065247885166
# 시간 : 4.496622323989868

# XGBClassifier
# pca = PCA(n_components=20)   
# 결과: 0.8855279123602661
# 시간 : 5.378031492233276