import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor

#1. 데이터
path = 'D:/study_data/_data/'
data = pd.read_csv(path + 'winequality-white.csv', index_col=None,
                   header=0, sep=';')  

# print(data.shape)   # (4898, 12)
# print(data.describe())  # 데이터의 내용들 나옴.
# print(data.info())

# pandas를 numpy로 바꾸는 법
data2 = data.to_numpy()
# data = data.values
# print(type(data))   # <class 'numpy.ndarray'>
# print(data.shape)   # (4898, 12)

x = data2[:, :11]
y = data2[:, 11]
# print(x.shape, y.shape) # (4898, 11) (4898,)

print(np.unique(y, return_counts=True))
# (array([3., 4., 5., 6., 7., 8., 9.]), array([  20,  163, 1457, 2198,  880,  175,    5], dtype=int64))
print(data['quality'].value_counts())

print(y[:20])
# [6. 6. 6. 6. 6. 6. 6. 6. 6. 6. 5. 5. 5. 7. 5. 7. 6. 8. 6. 5.]

newlist = []
for index, value in enumerate(y) :
    if value == 9 :
        y[index] = 7
    elif value == 8 :
        y[index] = 8
    elif value == 7 :
        y[index] = 7
    elif value == 6 :
        y[index] = 6
    elif value == 5 :
        y[index] = 5
    elif value == 4 :
        y[index] = 4
    elif value == 3 :
        y[index] = 4
    else:
        y[index] = 0                    
             
print(np.unique(newlist, return_counts=True))   
# (array([0, 1, 2]), array([1640, 2198, 1060], dtype=int64))
    


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)    


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True,
                                                    train_size=0.8, stratify=y)

scaler = PolynomialFeatures()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
from sklearn.ensemble import RandomForestClassifier
model = XGBClassifier(n_jobs = -1,
    n_estimators=1000,
    learning_rate = 0.1,
    max_depth = 6,
    min_child_weight = 1,
    subsample =1,
    colsample_bytree =1,
    reg_alpha =1,              
    reg_lambda=0,              
    tree_method= 'gpu_hist',
    predictor= 'gpu_predictor')

#3. 훈련
model.fit(x_train, y_train, verbose=1,
          eval_set=[(x_test, y_test)],
          eval_metric='mlogloss',          
          early_stopping_rounds=20)

#4. 평가,예측
y_predict = model.predict(x_test)

from sklearn.metrics import accuracy_score, f1_score    

score = model.score(x_test, y_test)
print('model.score : ', score)
print('acc_score : ', accuracy_score(y_test,y_predict))
print('f1_score(macro) : ', f1_score(y_test,y_predict, average='macro'))
print('f1_score(micro) : ', f1_score(y_test,y_predict, average='micro'))

# model.score :  0.6928571428571428
# acc_score :  0.6928571428571428
# f1_score(macro) :  0.6338260621355768
# f1_score(micro) :  0.6928571428571428


# 3,4 -> 0
# acc_score :  0.6928571428571428
# f1_score(macro) :  0.6338260621355768

# 5,6,7 -> 1
# acc_score :  0.9387755102040817
# f1_score(macro) :  0.5637499714943788

# 8,9 -> 2
# acc_score :  0.6642857142857143
# f1_score(macro) :  0.5487294434293664

# 3,4,5 -> 0
# acc_score :  0.6938775510204082
# f1_score(macro) :  0.6316156845104214

# 6 -> 1
# acc_score :  0.6642857142857143
# f1_score(macro) :  0.5354196180322566

# 7,8,9 -> 2
# acc_score :  0.6877551020408164
# f1_score(macro) :  0.6326322271427739