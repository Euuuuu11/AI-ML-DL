from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes, load_boston
from sklearn.metrics import mean_absolute_error, r2_score, median_absolute_error
import tensorflow as tf
import numpy as np
import pandas as pd
tf.compat.v1.set_random_seed(1234)

# 1. 데이터

datasets = load_boston()

x = datasets.data
y = datasets.target

print(x.shape, y.shape)
# (506, 13) (506,)

y = y.reshape(-1, 1)
# y = pd.get_dummies(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=123
)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 13])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([13, 1]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), name='bias')

hypothesis = tf.compat.v1.matmul(x, w) + b  # matmul :: 행렬곱 함수
# hypothesis = :: y의 shape값과 같아야한다.

loss = tf.reduce_mean(tf.square(hypothesis-y))  # mse
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(loss)

# 3-1. 컴파일

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epoch = 2001
for epochs in range(epoch):
    cost_val, hy_val, _ = sess.run([loss, hypothesis, train],
                                   feed_dict={x: x_train, y: y_train})
    if epochs % 20 == 0:
        print(epochs, "loss :: ", cost_val, "\n", hy_val)


# 4. 평가, 예측

r2 = r2_score(y_test, hy_val)
print("R2 :: ", r2)
mae = mean_absolute_error(y_test, hy_val)
print("mae :: ", mae)

sess.close()