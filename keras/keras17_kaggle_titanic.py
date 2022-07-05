# [타이타닉]
from joblib import parallel_backend
import pandas as pd
from sklearn import datasets
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.models import Sequential   
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error

#1. 데이터
path = './_data/kaggle_titanic/'
train_set = pd.read_csv(path + 'train.csv')
test_set = pd.read_csv(path + 'test.csv',index_col=0)
# print(train_set.shape, test_set.shape) # (891, 12) (418, 11)
# print(train_set.columns.values)
# ['PassengerId' 'Survived' 'Pclass' 'Name' 'Sex' 'Age' 'SibSp' 'Parch'    
# 'Ticket' 'Fare' 'Cabin' 'Embarked']
# print(train_set['Survived'].value_counts()) # 0 549 , 1 342

#. 결측치 처리
# print(train_set.isnull().sum())
# print(test_set.isnull().sum())

train_set=train_set.drop(columns='Cabin') # 'Cabin' 열 지움
test_set=test_set.drop(columns='Cabin')

train_set.loc[train_set['Sex']=='male', 'Sex']=0  # male = 0, female = 1
train_set.loc[train_set['Sex']=='female','Sex']=1 # 으로 인식 할 수 있게 인코딩
test_set.loc[test_set['Sex']=='male','Sex']=0
test_set.loc[test_set['Sex']=='female','Sex']=1
train_set['Sex']
# train_set 데이터에 있는 Pclass_3(없으면 만듬)
# train_set 데이터의 ['Pclass'] 열의 값이 3인 애들은 Tuer or False로 입력
train_set['Pclass_3']=(train_set['Pclass']==3)
train_set['Pclass_2']=(train_set['Pclass']==2)
train_set['Pclass_1']=(train_set['Pclass']==1)

test_set['Pclass_3']=(test_set['Pclass']==3)
test_set['Pclass_2']=(test_set['Pclass']==2)
test_set['Pclass_1']=(test_set['Pclass']==1)

train_set=train_set.drop(columns='Pclass') # 'Pclass' 열 삭제
test_set=test_set.drop(columns='Pclass')
# print(train_set.columns.values)  # 'Pclass' 열이 삭제되고, 'Pclass123'이 생긴 걸 볼 수 있다.
# ['PassengerId' 'Survived' 'Name' 'Sex' 'Age' 'SibSp' 'Parch' 'Ticket'
#  'Fare' 'Embarked' 'Pclass_3' 'Pclass_2' 'Pclass_1']

# 'Fare' 한개의 결측치를 0으로 채워줌
test_set.loc[test_set['Fare'].isnull(),'Fare']=0

# 'Age'라는 열은 별로 연관이 없을거 같아서 지워준다.
train_set=train_set.drop(columns='Age')
test_set=test_set.drop(columns='Age')

# 'Sibsp' = 형제자매+배우자, 'Parch' = 부모+자녀
# 'Sibsp' + 'Parch' + '1'(본인) = 'FamilySize' 
train_set['FamilySize']=train_set['SibSp']+train_set['Parch']+1
test_set['FamilySize']=test_set['SibSp']+test_set['Parch']+1
# print(train_set.head())

train_set['Single']=train_set['FamilySize']==1
train_set['Nuclear']=(2<=train_set['FamilySize']) & (train_set['FamilySize']<=4)
train_set['Big']=train_set['FamilySize']>=5

test_set['Single']=test_set['FamilySize']==1
test_set['Nuclear']=(2<=test_set['FamilySize']) & (test_set['FamilySize']<=4)
test_set['Big']=test_set['FamilySize']>=5
# print(train_set.head()) 
# 'Nuclear' 이란 가구가 생존율이 제일 높다. 그래서 나머지 열은 삭제하겠다.
train_set=train_set.drop(columns=['Single','Big','SibSp','Parch','FamilySize'])
test_set=test_set.drop(columns=['Single','Big','SibSp','Parch','FamilySize'])
# print(train_set.columns.values)

# 인코딩
train_set['EmbarkedC']=train_set['Embarked']=='C'
train_set['EmbarkedS']=train_set['Embarked']=='S'
train_set['EmbarkedQ']=train_set['Embarked']=='Q'
test_set['EmbarkedC']=test_set['Embarked']=='C'
test_set['EmbarkedS']=test_set['Embarked']=='S'
test_set['EmbarkedQ']=test_set['Embarked']=='Q'

# 원래 열 지움
train_set=train_set.drop(columns='Embarked')
test_set=test_set.drop(columns='Embarked')
# print(train_set.columns.values)


train_set=train_set.drop(columns='Name')
test_set=test_set.drop(columns='Name')

train_set=train_set.drop(columns='Ticket')
test_set=test_set.drop(columns='Ticket')

x = train_set.drop(['Survived'], axis=1).astype(float)
y = train_set['Survived'].astype(float)

print(x.shape, y.shape) # (891, 10) (891, )

x_train, x_test, y_train, y_test = train_test_split(x,y,
             train_size=0.8, shuffle=True, random_state=66)

#2. 모델구성
model = Sequential()
model.add(Dense(200, input_dim=10,activation="relu"))
model.add(Dense(100))
model.add(Dense(100,activation="relu"))
model.add(Dense(60))
model.add(Dense(60,activation="relu"))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=100, mode='min', 
              verbose=1, restore_best_weights=True) 

model.fit(x_train, y_train, epochs=100, batch_size=10, verbose=1, callbacks=[es],validation_split=0.2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
y_predict = model.predict(x_test)

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
y_predict = tf.argmax(y_predict,axis=1)
print(y_test,y_predict)
y_test = np.argmax(y_test,axis=1)


acc = accuracy_score(y_test, y_predict)
print('acc스코어 : ', acc)
# submission = pd.read_csv('C:\study\_data\kaggle_titanic\gender_submission.csv',index_col=0)
# submission['Survived'] = y_summit
# submission.to_csv('C:\study\_data\kaggle_titanic\gender_submission.csv', index=True)




















