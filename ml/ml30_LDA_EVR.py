import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, fetch_covtype, load_digits
from sklearn.decomposition import PCA 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xg
# print('xgboost버전 : ', xg.__version__) # xgboost버전 :  1.6.1

#1. 데이터
datasets = load_iris()          # (150, 2)->(150, 2)
datasets = load_breast_cancer() # (569, 30)->(569, 1)
datasets = load_wine()          # (178, 13)->(178, 2)
datasets = fetch_covtype()      # (581012, 54)->(581012, 6)
datasets = load_digits()        # (1797, 64)->(1797, 9)

x = datasets.data
y = datasets.target
print(x.shape)

lda = LinearDiscriminantAnalysis()
lda.fit(x,y)
x = lda.transform(x)
print(x.shape)


lda_EVR = lda.explained_variance_ratio_
cumsum = np.cumsum(lda_EVR)             
print(cumsum)


'''
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,shuffle=True,random_state=123,
                                                    stratify=y)

# lda = LinearDiscriminantAnalysis()
# lda.fit(x_train,y_train)
# x_train = lda.transform(x_train)
# x_test = lda.transform(x_test)

print(np.unique(y_train, return_counts=True))           
                                                            
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
'''