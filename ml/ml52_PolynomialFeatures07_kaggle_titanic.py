from cgi import test
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline

path = './_data/kaggle_titanic/'
train_set = pd.read_csv(path + 'train.csv',index_col =0)
test_set = pd.read_csv(path + 'test.csv', index_col=0)

##########전처리############
train_test_data = [train_set, test_set]
sex_mapping = {"male":0, "female":1}
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)

print(dataset)

for dataset in train_test_data:
    # 가족수 = 형제자매 + 부모님 + 자녀 + 본인
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    dataset['IsAlone'] = 1
    
    # 가족수 > 1이면 동승자 있음
    dataset.loc[dataset['FamilySize'] > 1, 'IsAlone'] = 0

for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
embarked_mapping = {'S':0, 'C':1, 'Q':2}
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)

for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract('([\w]+)\.', expand=False)
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].apply(lambda x: 0 if x=="Mr" else 1 if x=="Miss" else 2 if x=="Mrs" else 3 if x=="Master" else 4)

train_set['Cabin'] = train_set['Cabin'].str[:1]
for dataset in train_test_data:
    dataset['Age'].fillna(dataset.groupby("Title")["Age"].transform("median"), inplace=True)
for dataset in train_test_data:
    dataset['Agebin'] = pd.cut(dataset['Age'], 5, labels=[0,1,2,3,4])
for dataset in train_test_data:
    dataset["Fare"].fillna(dataset.groupby("Pclass")["Fare"].transform("median"), inplace=True)
for dataset in train_test_data:
    dataset['Farebin'] = pd.qcut(dataset['Fare'], 4, labels=[0,1,2,3])
    drop_column = ['Name', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin']

for dataset in train_test_data:
    dataset = dataset.drop(drop_column, axis=1, inplace=True)
print(train_set.head())


x = train_set.drop(['Survived'], axis=1,)
y = train_set['Survived']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=1234,
)

kfold = KFold(n_splits=5, shuffle=True, random_state=1234)

#2. 모델
model = make_pipeline(StandardScaler(),
                      LogisticRegression()
                      )
model.fit(x_train,y_train)

print('just score : ',model.score(x_test,y_test))
# 0.7665382927362877

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
print('just cv : ', scores)
print('just cv n/1 : ', np.mean(scores))

########################################## PolynomialFeatures 후 ##########################################
from sklearn.preprocessing import PolynomialFeatures

pf = PolynomialFeatures(degree=2, include_bias=False)
xp = pf.fit_transform(x)
# print(xp.shape)     # (506, 105)

x_train, x_test, y_train, y_test = train_test_split(
    xp, y, test_size=0.2, random_state=1234,
)
model = make_pipeline(StandardScaler(),
                      LogisticRegression()
                      )
model.fit(x_train,y_train)

print('poly score : ',model.score(x_test,y_test))

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
print('poly cv : ', scores)
print('poly cv n/1 : ', np.mean(scores))

# 0.8745129304823863

# just score :  0.8268156424581006
# just cv :  [0.77622378 0.74125874 0.79577465 0.84507042 0.83098592]
# just cv n/1 :  0.7978627006796021
# poly score :  0.8435754189944135
# poly cv :  [0.83216783 0.76923077 0.74647887 0.85915493 0.83802817]
# poly cv n/1 :  0.8090121146459175