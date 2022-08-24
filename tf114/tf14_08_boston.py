from sklearn.datasets import load_boston
import tensorflow as tf
tf.set_random_seed(66)

#1. 데이터
datasets = load_boston()
x_data = datasets.data
y_data = datasets.target
# print(x_data.shape, y_data.shape) # (506, 13) (506,
y_data = y_data.reshape(506,1)
# print(x_data.shape, y_data.shape) # (506, 13) (506, 1)


x = tf.placeholder(tf.float32, shape = [None,13])
y = tf.placeholder(tf.float32, shape = [None,1])

w = tf.compat.v1.Variable(tf.random.normal([13,1]), name = 'weight')     
b = tf.compat.v1.Variable(tf.random.normal([1,1]), name = 'bias')

#2. 모델구성
hypothesis =  tf.matmul(x,w) + b

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))    # mse

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)  
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epochs in range(200001):
    _, loss_val, w_val = sess.run([train, loss, w], feed_dict={x:x_data, y:y_data})
    if epochs % 200 ==0:
        print(epochs, '\t', loss_val)
    
#4. 예측
predict = tf.matmul(x,w_val) + b  
y_predict = sess.run(predict, feed_dict={x:x_data, y:y_data})
#print("예측 : " , y_predict)

from sklearn.metrics import r2_score, mean_absolute_error
r2 = r2_score(y_data, y_predict)
print('r2스코어 : ', r2)

sess.close()

# r2스코어 :  0.02379543779684523
