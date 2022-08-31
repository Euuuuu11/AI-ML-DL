import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder
import time
from sklearn.svm import LinearSVC, LinearSVR
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, accuracy_score


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
y = test_set['Survived']
y = np.array(y).reshape(-1, 1)


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,shuffle=True, random_state=9)
print(x_train.shape, x_test.shape) # (712, 7) (179, 7)
exit()

# 2. 모델 / sigmoid
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 7])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.zeros([7,80]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([80]), name='bias')
hidden = tf.compat.v1.nn.relu(tf.compat.v1.matmul(x, w) + b)

w = tf.compat.v1.Variable(tf.compat.v1.zeros([80,100]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([100]), name='bias')
hidden = tf.compat.v1.matmul(hidden, w) + b

w = tf.compat.v1.Variable(tf.compat.v1.zeros([100,90]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([90]), name='bias')
hidden = tf.compat.v1.nn.relu(tf.compat.v1.matmul(hidden, w) + b)

w = tf.compat.v1.Variable(tf.compat.v1.zeros([90,70]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([70]), name='bias')
hidden = tf.compat.v1.nn.relu(tf.compat.v1.matmul(hidden, w) + b)

w = tf.compat.v1.Variable(tf.compat.v1.zeros([70,50]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([50]), name='bias')
hidden = tf.compat.v1.nn.relu(tf.compat.v1.matmul(hidden, w) + b)

w = tf.compat.v1.Variable(tf.compat.v1.zeros([50,1]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), name='bias')
hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(hidden, w) + b)

# 3-1. 컴파일
loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis)) # binary_crossentropy

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-3)
optimizer = tf.train.AdamOptimizer(learning_rate=0.00000117)
train = optimizer.minimize(loss)

# 3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

epochs = 1001
for step in range(epochs):
    _, hy_val, cost_val, b_val = sess.run([train,hypothesis,loss,b], feed_dict={x:x_train, y:y_train})
    if step%20 == 0:
        print(step, cost_val, hy_val)
        
print('최종: ', cost_val, hy_val)

# 4. 평가, 예측
y_predict = sess.run(tf.cast(hy_val>=0.5, dtype=tf.float32))
acc = accuracy_score(y_train, y_predict)
print('acc: ', acc)

mae = mean_squared_error(y_train, hy_val)
print('mae: ', mae)

sess.close()

# acc:  0.6306179775280899
# mae:  0.24992371032655764