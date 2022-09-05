import tensorflow as tf
import keras
import numpy as np
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
tf.compat.v1.set_random_seed(123)

#1. 데이터
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.

#2. 모델구성
x = tf.compat.v1.placeholder(tf.float32, [None, 28, 28, 1])     # input_shape
y = tf.compat.v1.placeholder(tf.float32, [None, 10])     

w1 = tf.compat.v1.get_variable('w1', shape=[2, 2, 1 ,128])   # kernel_size, color, filter
L1 = tf.nn.conv2d(x, w1, strides=[1,1,1,1], padding='SAME')
L1 = tf.nn.relu(L1)
L1_maxpool = tf.nn.max_pool2d(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
# model.add(Conv2d(64, kernel_size=(2,2), input_shape =(28, 28, 1), activation = 'relu'))

# print(w1)   # <tf.Variable 'w1:0' shape=(2, 2, 1, 64) dtype=float32_ref>
# print(L1)   # Tensor("Conv2D:0", shape=(?, 28, 28, 128), dtype=float32)
# print(L1_maxpool)   # Tensor("MaxPool2d:0", shape=(?, 14, 14, 128), dtype=float32)

# Layer2 
w2 = tf.compat.v1.get_variable('w2', shape=[3, 3, 128 ,64])   # kernel_size, color, filter
L2 = tf.nn.conv2d(L1_maxpool, w2, strides=[1,1,1,1], padding='VALID')
L2 = tf.nn.selu(L2)
L2_maxpool = tf.nn.max_pool2d(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

# print(L2)           # Tensor("Selu:0", shape=(?, 13, 13, 64), dtype=float32)
# print(L2_maxpool)   # Tensor("MaxPool2d_1:0", shape=(?, 7, 7, 64), dtype=float32)   

# Layer3 
w3 = tf.compat.v1.get_variable('w3', shape=[3, 3, 64 ,32])   # kernel_size, color, filter
L3 = tf.nn.conv2d(L2_maxpool, w3, strides=[1,1,1,1], padding='VALID')
L3 = tf.nn.elu(L3)

# print(L3)   # Tensor("Elu:0", shape=(?, 4, 4, 32), dtype=float32)

# Flatten
L_flat = tf.reshape(L3, [-1, 4*4*32])
# print("플레튼 : ", L_flat)      # 플레튼 :  Tensor("Reshape:0", shape=(?, 512), dtype=float32)

# Layer4 DNN
w4 = tf.get_variable('w4', shape=[4*4*32 ,100],
                     initializer=tf.contrib.layers.xavier_initializer())   
b4 = tf.Variable(tf.random_normal([100], name = 'b4'))
L4 = tf.nn.selu(tf.matmul(L_flat, w4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=0.7)   # rate=0.3

# Layer5 DNN
w5 = tf.get_variable('w5', shape=[100 ,10],
                     initializer=tf.contrib.layers.xavier_initializer())   
b5 = tf.Variable(tf.random_normal([10], name = 'b5'))
L5 = tf.matmul(L4, w5) + b5
hypothesis = tf.nn.softmax(L5)

# print(hypothesis)   # Tensor("Softmax:0", shape=(?, 10), dtype=float32)

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
    _, hy_val, cost_val, b_val = sess.run([train,hypothesis,loss,b5], feed_dict={x:x_train, y:y_train})
    if step%20 == 0:
        print(step, cost_val, hy_val)
        
print('최종: ', cost_val, hy_val)

# 4. 평가, 예측
y_predict = sess.run(tf.cast(hy_val>=0.5, dtype=tf.float32))
acc = accuracy_score(y_train, y_predict)
print('acc: ', acc)

mae = mean_squared_error(y_train, hy_val)
print('mae: ', mae)
