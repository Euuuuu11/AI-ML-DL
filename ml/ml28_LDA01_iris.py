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
datasets = load_iris()
# datasets = load_breast_cancer()


x = datasets.data
y = datasets.target
# print(x.shape)              # (150, 4)

# le = LabelEncoder()
# y = le.fit_transform(y)

# pca = PCA(n_components=4)       #   54 >10
# x = pca.fit_transform(x)

# lda = LinearDiscriminantAnalysis()
lda = LinearDiscriminantAnalysis(n_components=2)
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

# pca = PCA(n_components=4)  
# 결과 : 0.87904570658096
# 시간 : 0.4218714237213135

# lda = LinearDiscriminantAnalysis(n_components=2)
# 결과 : 0.9508939352634385
# 시간 : 0.4242877960205078