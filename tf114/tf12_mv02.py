import tensorflow as tf
tf.compat.v1.set_random_seed(66)

x_data = [[73, 51, 65],
         [92, 98, 11],   
         [89, 31, 33],
         [99, 33, 100],
         [17, 66, 79]]
y_data = [[152],[185],[180],[205],[142]]

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])   # placeholder를 이용해서 입력값을 받을 수 있다.
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([3,1], name='weight'))
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1], name='bias'))

hypothesis = tf.compat.v1.matmul(x, w) + b  # matmul 행렬연산
# hypothesis 와 y의 shape값이 일치해야한다.

loss = tf.reduce_mean(tf.square(hypothesis-y_data))  # mse
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epoch = 2001
for epochs in range(epoch):
    cost_val, hy_val, _ = sess.run([loss, hypothesis, train], 
                                   feed_dict={x:x_data , y:y_data})
    if epochs % 20 ==0:
        print(epochs, "loss : ", cost_val, "\n", hy_val)
        
sess.close()

# y_predict = x_data * hy_v
# print(y_predict)

from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
r2 = r2_score(y_data, hy_val)
print('r2 : ', r2)

mae = mean_absolute_error(y_data, hy_val)  
print('mae : ', mae)



