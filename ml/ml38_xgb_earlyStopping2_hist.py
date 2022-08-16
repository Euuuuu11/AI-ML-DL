#[실습] girdSearchfrom sklearn.datasets import load_breast_cancer

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold,StratifiedKFold,train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
import time 
import numpy as np
#1. 데이터 
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

print(x.shape,y.shape)          # (569, 30) (569,)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.8, random_state=123, stratify=y)


scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5

kfold = StratifiedKFold(n_splits=n_splits ,shuffle=True, random_state=123)


#2. 모델 

model = XGBClassifier(random_state=123,                
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    gamma=1
            #         min_child_weight=1,
            #         subsample=1,
            #         colsample_bytree=1,
            #         colsample_bylevel=1,
            # #   'colsample_byload':[1],
            #         reg_alpha=0,
            #         reg_lambda=1
)

# model = GridSearchCV(xgb, parameters, cv =kfold, n_jobs=8)

import time
start = time.time()

model.fit(x_train,y_train, early_stopping_rounds =10,
          eval_set = [(x_train,y_train),(x_test,y_test)],   # 훈련 + 학습 # 뒤에걸 인지한다
        #   eval_set = [(x_test,y_test)]                 
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


# 결과 : 0.9736842105263158
# 시간 : 0.1349799633026123
# 최종 acc : 0.9736842105263158

print('=============================================')
hist =model.evals_result()
# hist = np.array(hist)


print(hist['validation_0']['error'])
print(hist['validation_1']['error'])



# [실습]
# 그래프 그리기 
import numpy as np
import matplotlib.pyplot as plt 

plt.subplot(1.5,1.5,1)
plt.plot(hist['validation_0']['error'])
# plt.plot([0.02857142857142857, 0.02197802197802198, 0.02197802197802198, 0.02197802197802198, 0.02417582417582418, 0.01538461538461539, 0.01978021978021978, 0.01538461538461539, 0.01318681318681319, 0.01318681318681319, 0.01318681318681319, 0.01318681318681319])

plt.subplot(1.5,1.5,1)
plt.plot(hist['validation_1']['error'])

# plt.plot([0.04385964912280702, 0.02631578947368421, 0.05263157894736842, 0.03508771929824561, 0.04385964912280702, 0.03508771929824561, 0.03508771929824561, 0.04385964912280702, 0.04385964912280702, 0.05263157894736842, 0.04385964912280702, 0.05263157894736842])
# plt.plot(hist['validation_0'])
# plt.plot(hist['validation_1'])

plt.xlabel('validation_0')
plt.ylabel('validation_1')
plt.show()






exit()
def plot_feature_importances(hist):
    n_features = hist.data.shape[1]
    plt.barh(np.arange(n_features),hist.feature_importances_, align ='center')
    plt.yticks(np.arange(n_features),datasets.feature_names)
    plt.xlabel('Feature Important')
    plt.ylabel('Features')
    plt.ylim(-1,n_features)
    plt.title(hist)
    
plot_feature_importances(hist)
plt.show() 