import numpy as np
import tensorflow as tf
tf.set_random_seed(123)

x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 6, 7]]
y_data = [[0, 0, 1],    # 2
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],    # 1
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],    # 0
          [1, 0, 0]]

#2. 모델구성 // 시작
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 4])   # placeholder를 이용해서 입력값을 받을 수 있다.

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([4, 3], name='weight'))

b = tf.Variable(tf.random_normal([1, 3], name='bias'))

y = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])

hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)  # matmul 행렬연산
# model.add(Dense(3, activation= "softmax", input_dim= 4))
# hypothesis 와 y의 shape값이 일치해야한다.

#3-1. 컴파일
# loss = tf.reduce_mean(tf.square(hypothesis-y_data))                   # mse
loss = -tf.reduce_sum(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))  # categorical_crossentropy
# model.compile(loss='categorical_crossentropy')

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# train = optimizer.minimize(loss)

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epoch = 2001
for epochs in range(epoch):
    cost_val, hy_val, _ = sess.run([loss, hypothesis, train], 
                                   feed_dict={x:x_data , y:y_data})
    if epochs % 20 ==0:
        print(epochs, "loss : ", cost_val, "\n", hy_val)
        

