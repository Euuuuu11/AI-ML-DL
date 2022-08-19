
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import pandas as pd
from sqlalchemy import true                                 # pandas : 엑셀땡겨올때 씀 python 지원하는 엑셀을 불러오는 기능.
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LogisticRegression, LinearRegression     # LogisticRegression 분류모델 LinearRegression 회귀
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 

#1. 데이터
path = './_data/ddarung/'
train_set = pd.read_csv(path + 'train.csv', 
                        index_col=0) 
test_set = pd.read_csv(path + 'test.csv', 
                       index_col=0)

train_set =  train_set.dropna()
test_set = test_set.fillna(test_set.mean())

x = train_set.drop(['count'], axis=1) 
y = train_set['count']



from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,
                                                    random_state=123,shuffle=True)


#2. 모델 
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBClassifier,XGBRFRegressor        # activate tf282gpu > pip install xgboost 

model1 = DecisionTreeRegressor()
model2 = RandomForestRegressor()
model3 = RandomForestRegressor()
model4 = XGBRFRegressor ()

#3. 훈련
model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)
model4.fit(x_train,y_train)

#4. 예측
result1 = model1.score(x_test,y_test)
print("model.score:",result1)

from sklearn.metrics import accuracy_score, r2_score

y_predict = model1.predict(x_test)
r2 = r2_score(y_test,y_predict)

print( 'r2_score1 :',r2)
print(model1,':',model1.feature_importances_)   # 중요한 피쳐를 구분하는 것 중요성이 떨어지는것을 버린다. 
print("===================================")

result2 = model2.score(x_test,y_test)
print("model1.score:",result2)


y_predict2 = model2.predict(x_test)
r2 = r2_score(y_test,y_predict2)

print( 'r2_score2 :',r2)
print(model2,':',model2.feature_importances_)   # 중요한 피쳐를 구분하는 것 중요성이 떨어지는것을 버린다. 
print("===================================")

result3 = model3.score(x_test,y_test)
print("model2.score3:",result3)


y_predict3 = model3.predict(x_test)
r2 = r2_score(y_test,y_predict3)

print( 'r2_score3 :',r2)
print(model3,':',model3.feature_importances_)   # 중요한 피쳐를 구분하는 것 중요성이 떨어지는것을 버린다. 
print("===================================")

result4 = model4.score(x_test,y_test)
print("model4.score:",result4)


y_predict4 = model4.predict(x_test)
r2 = r2_score(y_test,y_predict4)

print( 'r2_score4 :',r2)
print(model4,':',model4.feature_importances_)   # 중요한 피쳐를 구분하는 것 중요성이 떨어지는것을 버린다. 
print("===================================")

# 삭제후 

# model.score: 0.6300146059727504
# r2_score1 : 0.6300146059727504
# DecisionTreeRegressor() : [0.58889837 0.19193815 0.07812078 0.0589318  0.04272128 0.03938963]
# ===================================
# model1.score: 0.7821477073674483
# r2_score2 : 0.7821477073674483
# RandomForestRegressor() : [0.59613307 0.19878038 0.06884772 0.05845012 0.04331326 0.03447545]
# ===================================
# model2.score3: 0.7953047052028843
# r2_score3 : 0.7953047052028843   
# RandomForestRegressor() : [0.59692309 0.19396559 0.06771564 0.05766864 0.04795155 0.03577549]
# ===================================
# model4.score: 0.7438176570900675
# r2_score4 : 0.7438176570900675
# XGBRFRegressor


# 삭제전 
# model.score: 0.6203555250587188
# r2_score1 : 0.6203555250587188
# DecisionTreeRegressor() : [0.58266757 0.17451747 0.02718274 0.01550531 0.02803954 0.04941452  
#  0.04749699 0.04161734 0.03355851]
# ===================================
# model1.score: 0.7961345753061605
# r2_score2 : 0.7961345753061605
# RandomForestRegressor() : [0.58372849 0.18168007 0.02193944 0.03235003 0.03393578 0.04210459  
#  0.04164655 0.03494641 0.02766864]
# ===================================
# model2.score3: 0.7954735396341172
# r2_score3 : 0.7954735396341172
# RandomForestRegressor() : [0.57913503 0.19415576 0.0211313  0.03100085 0.03570498 0.04025897  
#  0.03780124 0.0357234  0.02508846]
# ===================================
# model4.score: 0.7576400722892229
# r2_score4 : 0.7576400722892229
# XGBRFRegressor



