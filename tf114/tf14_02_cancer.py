import tensorflow as tf
tf.compat.v1.set_random_seed(123)
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


#1. 데이터
datasets = load_breast_cancer()
x,y = datasets.data, datasets.target
# print(x_data.shape, y_data.shape)     # (569, 30) (569,)

y = y.reshape(-1, 1)
# print(x_data.shape, y_data.shape)     # (569, 30) (569, 1)

#[실습] 시그모이드 빼고 만들기.

x_train, x_test, y_train, y_test =train_test_split(
    x, y, test_size=0.2, random_state=123, stratify = y) 

# print(type(x_train), type(y_train))     # <class 'numpy.ndarray'> <class 'numpy.ndarray'>
# print(x_train.dtype, y_train.dtype)     # float64 int32

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 30])   # placeholder를 이용해서 입력값을 받을 수 있다.
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([30,1], name='weight'))
b = tf.compat.v1.Variable(tf.zeros([1], name='bias'))

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
hypothesis = tf.compat.v1.sigmoid(tf.matmul(x, w) + b)  # matmul 행렬연산
# model.add(Dense(1, activation= "sigmoid", input_dim= 2))
# hypothesis 와 y의 shape값이 일치해야한다.

#3-1. 컴파일
# loss = tf.reduce_mean(tf.square(hypothesis-y_data))                   # mse
loss = -tf.reduce_sum(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))  # binary_crossentropy

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
# optimizer = tf.train.AdamOptimizer(learning_rate=1e-5)
# model.compile(loss='binary_crossentropy')

train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epoch = 101
for epochs in range(epoch):
    cost_val, hy_val, _ = sess.run([loss, hypothesis, train], 
                                   feed_dict={x:x_train , y:y_train})
    if epochs % 20 ==0:
        print(epochs, "loss : ", cost_val, "\n", hy_val)
        
# exit()
#4. 평가, 예측
y_predict = sess.run(tf.cast(hy_val>0.5, dtype=tf.float32))

sess.close()

from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
acc = accuracy_score(y_test, y_predict)
print('acc : ', acc)

mae = mean_absolute_error(y_test, y_predict)  
print('mae : ', mae)