from sklearn.datasets import fetch_california_housing
import tensorflow as tf
tf.set_random_seed(66)

#1. 데이터
datasets = fetch_california_housing()
x_data = datasets.data
y_data = datasets.target

# print(x_data.shape, y_data.shape)   # (20640, 8) (20640,)
# exit()
y_data = y_data.reshape(20640,1)
# print(x_data.shape, y_data.shape) # (506, 13) (506,)


x = tf.placeholder(tf.float32, shape = [None,8])
y = tf.placeholder(tf.float32, shape = [None,1])

w = tf.compat.v1.Variable(tf.random.normal([8,1]), name = 'weight')     
b = tf.compat.v1.Variable(tf.random.normal([1]), name = 'bias')

#2. 모델구성
hypothesis = tf.compat.v1.sigmoid(tf.matmul(x, w) + b)

#3-1. 컴파일
loss = -tf.reduce_sum(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))    # mse

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)  
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epochs in range(20001):
    _, loss_val, w_val = sess.run([train, loss, w], feed_dict={x:x_data, y:y_data})
    if epochs % 200 ==0:
        print(epochs, '\t', loss_val)
    
#4. 예측
predict = tf.matmul(x,w_val) + b  
y_predict = sess.run(predict, feed_dict={x:x_data, y:y_data})
# print("예측 : " , y_predict)

from sklearn.metrics import r2_score, mean_absolute_error
r2 = r2_score(y_data, y_predict)
print('r2스코어 : ', r2)

sess.close()
# r2스코어 :  0.5766435104265699