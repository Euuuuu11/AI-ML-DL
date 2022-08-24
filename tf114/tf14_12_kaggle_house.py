import tensorflow as tf
tf.set_random_seed(66)
import pandas as pd

#1. 데이터
path = 'C:\study\_data\kaggle_house/'   

train = pd.read_csv(path + 'train.csv')
test_file = pd.read_csv(path + 'test.csv')  
submit_file = pd.read_csv(path + 'sampleSubmission.csv')

x_data = train.drop(['datetime', 'casual','registered', 'count'], axis=1) 
#print(x.columns)

test_file = test_file.drop(['datetime'], axis=1)

#print(x.shape)    # (10886, 8)

y_data = train['count']
#print(y.shape)  #(10886,)
y_data = y_data.values.reshape(10886,1)

x = tf.placeholder(tf.float32, shape = [None,8])
y = tf.placeholder(tf.float32, shape = [None,1])

w = tf.compat.v1.Variable(tf.random.normal([8,1]), name = 'weight')     
b = tf.compat.v1.Variable(tf.random.normal([1]), name = 'bias')

#2. 모델구성
hypothesis =  tf.matmul(x,w) + b 

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))    # mse

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-6)  
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epochs in range(20001):
    _, loss_val, w_val = sess.run([train, loss, w], feed_dict={x:x_data, y:y_data})
    print(epochs, '\t', loss_val, '\t', w_val)
    
#4. 예측
predict = tf.matmul(x,w_val) + b  
y_predict = sess.run(predict, feed_dict={x:x_data, y:y_data})
print("예측 : " , y_predict)

from sklearn.metrics import r2_score, mean_absolute_error
r2 = r2_score(y_data, y_predict)
print('r2스코어 : ', r2)

sess.close()
''' 
r2스코어 :  0.2160568946163962
'''