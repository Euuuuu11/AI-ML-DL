
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import tensorflow as tf
from sklearn.metrics import r2_score
from sklearn.svm import LinearSVC, LinearSVR
import time
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/gulim.TTc"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)


# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

y = y.reshape(-1, 1)


# y = pd.get_dummies(y)


print(x.shape, y.shape)
# (20640, 8) (20640, 1)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=123
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


x = tf.compat.v1.placeholder(tf.float32, shape=[None, 8])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([8, 1]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), name='bias')

hypothesis = tf.compat.v1.matmul(x, w) + b  # matmul :: 행렬곱 함수
# hypothesis = :: y의 shape값과 같아야한다.

loss = tf.reduce_mean(tf.square(hypothesis-y))  # mse
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
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
y_predict = sess.run(hypothesis, feed_dict={x: x_test, y: y_test})

r2 = r2_score(y_test, y_predict)
print("R2 :: ", r2)

sess.close()

# R2 ::  0.3920209994306497