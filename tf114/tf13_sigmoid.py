import tensorflow as tf
tf.compat.v1.set_random_seed(123)

#1. 데이터
x_data = [[1,2], [2,3], [3,1], [4,3], [5,3], [6,2]]     # (6,2) 
y_data = [[0], [0], [0], [1], [1], [1]]                 # (6,1)    

#[실습] 시그모이드 빼고 만들기.

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])   # placeholder를 이용해서 입력값을 받을 수 있다.
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([2,1], name='weight'))
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1], name='bias'))

#2. 모델
hypothesis = tf.compat.v1.sigmoid(tf.matmul(x, w) + b)  # matmul 행렬연산
# model.add(Dense(1, activation= "sigmoid", input_dim= 2))
# hypothesis 와 y의 shape값이 일치해야한다.

#3-1. 컴파일
# loss = tf.reduce_mean(tf.square(hypothesis-y_data))                   # mse
loss = -tf.reduce_sum(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))  # binary_crossentropy

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
# model.compile(loss='binary_crossentropy')

train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epoch = 2001
for epochs in range(epoch):
    cost_val, hy_val, _ = sess.run([loss, hypothesis, train], 
                                   feed_dict={x:x_data , y:y_data})
    if epochs % 20 ==0:
        print(epochs, "loss : ", cost_val, "\n", hy_val)
        

#. 평가, 예측
y_predict = sess.run(tf.cast(hy_val>0.5, dtype=tf.float32))

sess.close()

from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
acc = accuracy_score(y_data, y_predict)
print('acc : ', acc)

mae = mean_absolute_error(y_data, y_predict)  
print('mae : ', mae)