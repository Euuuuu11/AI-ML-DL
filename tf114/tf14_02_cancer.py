from sklearn.metrics import r2_score, accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import mean_absolute_error, r2_score, median_absolute_error
import tensorflow as tf
tf.compat.v1.set_random_seed(1234)

# 1. 데이터

datasets = load_breast_cancer()

x = datasets.data
y = datasets.target

print(x.shape, y.shape)
# (442, 10) (442,)

y = y.reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=123
)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([10, 1]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), name='bias')

hypothesis = tf.compat.v1.matmul(x, w) + b  # matmul :: 행렬곱 함수
# hypothesis = :: y의 shape값과 같아야한다.

loss = tf.reduce_mean(tf.square(hypothesis-y))  # mse
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

# 3-1. 컴파일

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epoch = 1001
for epochs in range(epoch):
    cost_val, hy_val, _ = sess.run([loss, hypothesis, train],
                                   feed_dict={x: x_train, y: y_train})
    if epochs % 20 == 0:
        print(epochs, "loss :: ", cost_val, "\n", hy_val)


# 4. 평가, 예측

y_predict = sess.run(hypothesis, feed_dict={x: x_test, y: y_test})
r2 = r2_score(y_test, y_predict)
print('r2 :', r2)
sess.close()

# r2 : 0.1090695286114044