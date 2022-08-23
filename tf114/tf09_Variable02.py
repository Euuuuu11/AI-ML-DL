import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

tf.compat.v1.set_random_seed(123)

# [실습]
# 08_2 카피

# 실습
# lr 수정해서 epoch를 100번 이하로 줄인다.
# step = 100 이하, w = 1.99, b = 0.99

x_train_data = [1, 2, 3]
y_train_data = [3, 5, 7]

# 1. 데이터
# x = [1, 2, 3, 4, 5]
# y = [1, 2, 3, 4, 5]

x_train = tf.placeholder(tf.float32, shape=[None])
y_train = tf.placeholder(tf.float32, shape=[None])

W = tf.Variable(tf.random_normal([1]), dtype=tf.float32)  # random_normal = 갯수
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)

# 2. 모델구성
hypothesis = x_train * W + b  # y = wx + b

# 3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis-y_train))  # mse
# square = 제곱, reduce_mean = 평균
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.10)
train = optimizer.minimize(loss)    # loss 값이 제일 최소값이 되는걸 찾아낸다.
# model.compile(loss = 'mse', optimizer = 'sgd')

############################# 1. Session() // sees.run(변수) #############################
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())  # 초기화 시킨다.

epochs = 100
for step in range(epochs):
    # sess.run(train)
     _, loss_val, W_val, b_val = sess.run([train, loss, W, b],
                                             feed_dict=(
                                                 {x_train: x_train_data, y_train: y_train_data})
                                             )
    #  if step % 20 == 0:  # %20 의 나머지가 0이 아닐때 프린트함. 즉, 20번에 한번씩 실행시킨다.
    #     print(step, loss_val, W_val, b_val)
x_test_data = [6, 7, 8]
x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])
y_preidct = x_test * W_val + b_val                          # y_predict = model.predict(x_test)
print('1. [6, 7, 8] 예측 : ', sess.run(y_preidct, feed_dict={x_test:x_test_data}))

sess.close()  


############################# 2. Session() // 변수,eval(session = sess) #############################
# 3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis-y_train))  # mse
# square = 제곱, reduce_mean = 평균
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.10)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 100
for step in range(epochs):
    # sess.run(train)
     _, loss_val, W_val, b_val = sess.run([train, loss, W, b],
                                             feed_dict=(
                                                 {x_train: x_train_data, y_train: y_train_data})
                                             )
#  if step % 20 == 0:  # %20 의 나머지가 0이 아닐때 프린트함. 즉, 20번에 한번씩 실행시킨다.
#     print(step, loss_val, W_val, b_val)
x_test_data = [6, 7, 8]
x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])
y_preidct = x_test * W_val + b_val                          # y_predict = model.predict(x_test)
print('2. [6, 7, 8] 예측 : ', y_preidct.eval(session = sess, feed_dict={x_test:x_test_data}))

sess.close()  


############################# 3. InteractiveSession() // 변수.eval() #############################
# 3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis-y_train))  # mse
# square = 제곱, reduce_mean = 평균
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.10)
train = optimizer.minimize(loss)

sess = tf.compat.v1.InteractiveSession()    # InteractiveSession는 eval 안에 파라미터를 지정하지 않아도 된다.
sess.run(tf.compat.v1.global_variables_initializer())# 초기화 시킨다.

epochs = 100
for step in range(epochs):
    # sess.run(train)
     _, loss_val, W_val, b_val = sess.run([train, loss, W, b],
                                             feed_dict=(
                                                 {x_train: x_train_data, y_train: y_train_data})
                                             )
    #  if step % 20 == 0:  # %20 의 나머지가 0이 아닐때 프린트함. 즉, 20번에 한번씩 실행시킨다.
    #     print(step, loss_val, W_val, b_val)
x_test_data = [6, 7, 8]
x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])
y_preidct = x_test * W_val + b_val                          # y_predict = model.predict(x_test)
print('3. [6, 7, 8] 예측 : ', y_preidct.eval(feed_dict={x_test:x_test_data}))

sess.close()  