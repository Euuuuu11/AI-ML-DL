# y = wx + b
import tensorflow as tf
tf.set_random_seed(123)

#1. 데이터
x = [1, 2, 3, 4, 5]
y = [1, 2, 3, 4, 5]

w = tf.Variable(11,  dtype=tf.float32)
b = tf.Variable(10,  dtype=tf.float32)

#2. 모델 구성
hypothesis = x * w + b  # hypothesis 가설 / y = wx + b

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))    # mse / 제곱하고 평균을 구한다.
# square 제곱, reduce_mean 평균

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)
# model.compile(loss='mse', optimizer='sgd')

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

epochs = 5040
for step in range(epochs) : 
    sess.run(train)     # model.fit 부분
    if step %20 == 0:
        print(step, sess.run(loss), sess.run(w), sess.run(b))

sess.close()  