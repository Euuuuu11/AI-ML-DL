from time import time
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np 

path = 'D:\study_data\_data/dacon2/'
train_set = pd.read_csv(path + 'train.csv',
                        index_col=0)
test_set = pd.read_csv(path + 'test.csv', 
                       index_col=0)
# print(train_set[train_set['ProdTaken'].notnull()].groupby(['ProductPitched'])['ProdTaken'].mean())
# print(train_set['ProductPitched'].value_counts())
# print(train_set[train_set['ProdTaken'].notnull()].groupby(['OwnCar'])['ProdTaken'].mean())
# print(train_set['OwnCar'].value_counts())

train_set['TypeofContact'].fillna('Self Enquiry', inplace=True)
test_set['TypeofContact'].fillna('Self Enquiry', inplace=True)
train_set['Age'].fillna(train_set.groupby('Designation')['Age'].transform('mean'), inplace=True)
test_set['Age'].fillna(test_set.groupby('Designation')['Age'].transform('mean'), inplace=True)
train_set['Age']=np.round(train_set['Age'],0).astype(int)
test_set['Age']=np.round(test_set['Age'],0).astype(int)

train_set['MonthlyIncome'].fillna(train_set.groupby('Designation')['MonthlyIncome'].transform('mean'), inplace=True)
test_set['MonthlyIncome'].fillna(test_set.groupby('Designation')['MonthlyIncome'].transform('mean'), inplace=True)

train_set['NumberOfChildrenVisiting'].fillna(train_set.groupby('MaritalStatus')['NumberOfChildrenVisiting'].transform('mean'), inplace=True)
test_set['NumberOfChildrenVisiting'].fillna(test_set.groupby('MaritalStatus')['NumberOfChildrenVisiting'].transform('mean'), inplace=True)
train_set['NumberOfFollowups'].fillna(train_set.groupby('NumberOfChildrenVisiting')['NumberOfFollowups'].transform('mean'), inplace=True)
test_set['NumberOfFollowups'].fillna(test_set.groupby('NumberOfChildrenVisiting')['NumberOfFollowups'].transform('mean'), inplace=True)

train_set['DurationOfPitch']=train_set['DurationOfPitch'].fillna(0)
test_set['DurationOfPitch']=test_set['DurationOfPitch'].fillna(0)
# print(train_set[train_set['DurationOfPitch'].notnull()].groupby(['NumberOfChildrenVisiting'])['DurationOfPitch'].mean())

train_set['PreferredPropertyStar'].fillna(train_set.groupby('Occupation')['PreferredPropertyStar'].transform('mean'), inplace=True)
test_set['PreferredPropertyStar'].fillna(test_set.groupby('Occupation')['PreferredPropertyStar'].transform('mean'), inplace=True)

combine = [train_set,test_set]
for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 20, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 20) & (dataset['Age'] <= 29), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 29) & (dataset['Age'] <= 39), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 39) & (dataset['Age'] <= 49), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 49) & (dataset['Age'] <= 59), 'Age'] = 4
    dataset.loc[ dataset['Age'] > 59, 'Age'] = 5

train_set['NumberOfTrips'].fillna(train_set.groupby('DurationOfPitch')['NumberOfTrips'].transform('mean'), inplace=True)
test_set['NumberOfTrips'].fillna(test_set.groupby('DurationOfPitch')['NumberOfTrips'].transform('mean'), inplace=True)

train_set.loc[ train_set['Occupation'] =='Free Lancer' , 'Occupation'] = 'Salaried'
test_set.loc[ test_set['Occupation'] =='Free Lancer' , 'Occupation'] = 'Salaried'

train_set.loc[ train_set['Gender'] =='Fe Male' , 'Gender'] = 'Female'
test_set.loc[ test_set['Gender'] =='Fe Male' , 'Gender'] = 'Female'
cols = ['TypeofContact','Occupation','Gender','ProductPitched','MaritalStatus','Designation']
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm_notebook

for col in tqdm_notebook(cols):
    le = LabelEncoder()
    train_set[col]=le.fit_transform(train_set[col])
    test_set[col]=le.fit_transform(test_set[col])
               
x = train_set.drop(['ProdTaken',
                          'NumberOfChildrenVisiting',
                          'NumberOfPersonVisiting',
                          'OwnCar', 
                          'MonthlyIncome', 
                          'NumberOfFollowups',
                          ], axis=1)

test_set = test_set.drop(['NumberOfChildrenVisiting',
                          'NumberOfPersonVisiting',
                          'OwnCar', 
                          'MonthlyIncome', 
                          'NumberOfFollowups',
                          ], axis=1)
y = train_set['ProdTaken']
# print(x.shape) #1911,13

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.91,shuffle=True,random_state=1234,stratify=y)

from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

n_splits = 6
kfold = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=72)

cat_paramets = {"learning_rate" : [0.01],
                'depth' : [8],
                'od_pval' : [0.12673190617341812],
                # 'model_size_reg': [0.44979263197508923],
                'fold_permutation_block': [142],
                'l2_leaf_reg' :[0.33021257848638497]}
cat = CatBoostClassifier(random_state=72,verbose=False,n_estimators=1304)
model = RandomizedSearchCV(cat,cat_paramets,cv=kfold,n_jobs=-1,)

import time 
start_time = time.time()
model.fit(x_train,y_train)   
end_time = time.time()-start_time 
y_predict = model.predict(x_test)
results = accuracy_score(y_test,y_predict)

model.fit(x,y)   
y_submit = model.predict(test_set)
y_submit = np.round(y_submit,0)
submission = pd.read_csv(path + 'sample_submission.csv',
                      )
submission['ProdTaken'] = y_submit

submission.to_csv(path+'sample_submission_3.csv',index=False)





