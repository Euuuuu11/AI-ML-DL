import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import VotingClassifier, VotingRegressor   # 투표를 통해 최종 예측 결과를 결정하는 방식
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_diabetes
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm_notebook
from sklearn.preprocessing import LabelEncoder

#1. 데이터
path = 'C:\study\_data\kaggle_house/' # ".은 현재 폴더"
train_set = pd.read_csv(path + 'train.csv',
                        index_col=0)
test_set = pd.read_csv(path + 'test.csv', #예측에서 쓸거야!!
                       index_col=0)
drop_cols = ['Alley', 'PoolQC', 'Fence', 'MiscFeature']
test_set.drop(drop_cols, axis = 1, inplace =True)
submission = pd.read_csv(path + 'sample_submission.csv',#예측에서 쓸거야!!
                       index_col=0)
print(train_set)

print(train_set.shape) #(1459, 10)

train_set.drop(drop_cols, axis = 1, inplace =True)
cols = ['MSZoning', 'Street','LandContour','Neighborhood','Condition1','Condition2',
                'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Foundation',
                'Heating','GarageType','SaleType','SaleCondition','ExterQual','ExterCond','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
                'BsmtFinType2','HeatingQC','CentralAir','Electrical','KitchenQual','Functional',
                'FireplaceQu','GarageFinish','GarageQual','GarageCond','PavedDrive','LotShape',
                'Utilities','LandSlope','BldgType','HouseStyle','LotConfig']

for col in tqdm_notebook(cols):
    le = LabelEncoder()
    train_set[col]=le.fit_transform(train_set[col])
    test_set[col]=le.fit_transform(test_set[col])


###### 결측치 처리 1.제거##### dropna 사용
print(train_set.isnull().sum()) #각 컬럼당 결측치의 합계
train_set = train_set.fillna(train_set.median())
print(train_set.isnull().sum())
print(train_set.shape)
test_set = test_set.fillna(test_set.median())

x = train_set.drop(['SalePrice'],axis=1) #axis는 컬럼 
# print(x.columns)
y = train_set['SalePrice']
y = np.array(y)
y = y.reshape(-1, 1)
print(x.shape, y.shape) # (1460, 75) (1460, 1)
# exit()

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1234)
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 75])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.zeros([75,10]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([10]), name='bias')
hidden = tf.compat.v1.matmul(x, w) + b

w = tf.compat.v1.Variable(tf.compat.v1.zeros([10,30]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([30]), name='bias')
hidden = tf.compat.v1.matmul(hidden, w) + b

w = tf.compat.v1.Variable(tf.compat.v1.zeros([30,30]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([30]), name='bias')
hidden = tf.compat.v1.matmul(hidden, w) + b

w = tf.compat.v1.Variable(tf.compat.v1.zeros([30,30]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([30]), name='bias')
hidden = tf.compat.v1.matmul(hidden, w) + b

w = tf.compat.v1.Variable(tf.compat.v1.zeros([30,50]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([50]), name='bias')
hidden = tf.compat.v1.matmul(hidden, w) + b

w = tf.compat.v1.Variable(tf.compat.v1.zeros([50,20]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([20]), name='bias')
hidden = tf.compat.v1.matmul(hidden, w) + b

w = tf.compat.v1.Variable(tf.compat.v1.zeros([20,1]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), name='bias')
hypothesis = tf.compat.v1.matmul(hidden, w) + b


# 3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis-y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-8)
train = optimizer.minimize(loss)

# 3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

epochs = 2001
for step in range(epochs):
    _, hy_val, cost_val, b_val = sess.run([train,hypothesis,loss,b], feed_dict={x:x_train, y:y_train})
    if step%20 == 0:
        print(step, cost_val, hy_val)
        
print('최종: ', cost_val, hy_val)

y_pred = sess.run(hypothesis, feed_dict={x:x_test, y:y_test})

r2 = r2_score(y_test, y_pred)
print('r2: ', r2)

mae = mean_absolute_error(y_test, y_pred)
print('mae: ', mae)

# r2:  -6.264111566078972
# mae:  176339.98934586457