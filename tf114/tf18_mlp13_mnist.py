import tensorflow as tf
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.datasets import mnist

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
y_train = y_train.reshape(60000, 1)
y_test = y_test.reshape(10000, 1)
# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 / sigmoid
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 784])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.zeros([784,128]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([128]), name='bias')
hidden = tf.compat.v1.nn.relu(tf.compat.v1.matmul(x, w) + b)

w = tf.compat.v1.Variable(tf.compat.v1.zeros([128,10]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([10]), name='bias')
hidden = tf.compat.v1.matmul(hidden, w) + b

w = tf.compat.v1.Variable(tf.compat.v1.zeros([10,1]), name='weight')
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

# acc:  0.11236666666666667
# mae:  23.979021