from sklearn.datasets import load_wine
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
tf.set_random_seed(66)

#1. 데이터
dataset = load_wine()
x_data = dataset.data
y_data = dataset.target.reshape(-1,1)

ohe = OneHotEncoder()
ohe.fit(y_data)
y_data = ohe.transform(y_data).toarray()

# print(x_data.shape)  # (178, 13)
# print(y_data.shape)  # (178, 3)

x_train, x_test, y_train, y_test = train_test_split (x_data, y_data, train_size = 0.8,random_state=66)

x = tf.compat.v1.placeholder('float',shape=[None,13])
y = tf.compat.v1.placeholder('float',shape=[None,3])

w = tf.compat.v1.Variable(tf.zeros([13,3]), name='weight')
b = tf.compat.v1.Variable(tf.zeros([1,3]), name = 'bias')

#2. 모델링
hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)

#3. 컴파일, 훈련
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1)) # categorical_crossentropy

#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000001).minimize(loss)
optimizer = tf.train.AdamOptimizer(learning_rate=0.000001).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        _, loss_val, = sess.run([optimizer, loss], feed_dict = {x:x_train,y:y_train})
        if step % 200 ==0:
            print(step, loss_val)

    y_acc_test = sess.run(tf.argmax(y_test, 1))  # 실제값
    predict = sess.run(tf.argmax(sess.run(hypothesis, feed_dict={x:x_test}), 1))  # 예측값
    accuracy = accuracy_score(y_acc_test, predict)
    print("accuracy_score : ", accuracy)
    
# accuracy_score :  0.5833333333333334