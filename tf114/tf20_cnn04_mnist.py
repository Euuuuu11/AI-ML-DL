import tensorflow as tf
import keras
import numpy as np
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
tf.compat.v1.set_random_seed(123)
# 즉시 실행모드 : 2점대 즉시 실행
tf.compat.v1.disable_eager_execution() 


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
L1_maxpool = tf.nn.max_pool2d(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# model.add(Conv2d(64, kernel_size=(2,2), input_shape =(28, 28, 1), activation = 'relu'))

# print(w1)   # <tf.Variable 'w1:0' shape=(2, 2, 1, 64) dtype=float32_ref>
# print(L1)   # Tensor("Conv2D:0", shape=(?, 28, 28, 128), dtype=float32)
# print(L1_maxpool)   # Tensor("MaxPool2d:0", shape=(?, 14, 14, 128), dtype=float32)

# Layer2 
w2 = tf.compat.v1.get_variable('w2', shape=[3, 3, 128 ,64])   # kernel_size, color, filter
L2 = tf.nn.conv2d(L1_maxpool, w2, strides=[1,1,1,1], padding='VALID')
L2 = tf.nn.selu(L2)
L2_maxpool = tf.nn.max_pool2d(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# print(L2)           # Tensor("Selu:0", shape=(?, 13, 13, 64), dtype=float32)
# print(L2_maxpool)   # Tensor("MaxPool2d_1:0", shape=(?, 7, 7, 64), dtype=float32)   

# Layer3 
w3 = tf.compat.v1.get_variable('w3', shape=[3, 3, 32 ,32])   # kernel_size, color, filter
L3 = tf.nn.conv2d(L2_maxpool, w3, strides=[1,1,1,1], padding='VALID')
L3 = tf.nn.elu(L3)

# print(L3)   # Tensor("Elu:0", shape=(?, 4, 4, 32), dtype=float32)

# Flatten
L_flat = tf.reshape(L3, [-1, 4*4*32])
# print("플레튼 : ", L_flat)      # 플레튼 :  Tensor("Reshape:0", shape=(?, 512), dtype=float32)

# Layer4 DNN
w4 = tf.compat.v1.get_variable('w4', shape=[4*4*32 ,100])
                    #  initializer=tf.contrib.layers.xavier_initializer())   
b4 = tf.Variable(tf.random.normal([100], name = 'b4'))
L4 = tf.nn.selu(tf.matmul(L_flat, w4) + b4)
# L4 = tf.nn.dropout(L4, keep_prob=0.7)   # rate=0.3
L4 = tf.nn.dropout(L4, rate=0.3)   # rate=0.3

# Layer5 DNN
w5 = tf.compat.v1.get_variable('w5', shape=[100 ,10],)
                    #  initializer=tf.contrib.layers.xavier_initializer())   
b5 = tf.Variable(tf.compat.v1.random.normal([10], name = 'b5'))
L5 = tf.matmul(L4, w5) + b5
hypothesis = tf.nn.softmax(L5)

# print(hypothesis)   # Tensor("Softmax:0", shape=(?, 10), dtype=float32)

# 3-1. 컴파일
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=y)) 
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)


sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

#4. 훈련
training_epochs = 30
batch_size = 100
total_batch = int(len(x_train)/batch_size)
print(total_batch)


for epochs in range(training_epochs):   # 총 30번 돈다
    avg_loss = 0
    for i in range(total_batch):    # 총 600번 돈다
        start = i * batch_size      # 0
        end = start + batch_size    # 100
        batch_x, batch_y = x_train[start:end], y_train[start:end]       # 0~100
        
        feed_dict = {x:batch_x, y:batch_y}
        
        batch_loss,_ = sess.run([loss, optimizer], feed_dict=feed_dict)        
        avg_loss += batch_loss / total_batch
    print('Epoch : ', '%04d' %(epochs +1), 'loss : {:.9f}'.format(avg_loss))    
print('훈련 끝')


# exit()
prediction = tf.equal(tf.compat.v1.arg_max(hypothesis, 1), tf.compat.v1.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
print('ACC : ', sess.run(accuracy, feed_dict={x:x_test, y:y_test}))

