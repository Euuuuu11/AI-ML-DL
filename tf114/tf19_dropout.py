import tensorflow as tf

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w1 = tf.compat.v1.Variable(tf.random.normal([2, 30], name = 'weights1'))
b1 = tf.compat.v1.Variable(tf.random.normal([30], name = 'bias1'))

Hidden_layer1 = tf.compat.v1.sigmoid(tf.matmul(x, w1) + b1)
# model.add(Dense(30, input_shape=(2,), activation = 'sigmoid'))

dropout_layers = tf.compat.v1.nn.dropout(Hidden_layer1, keep_prob=0.3)

# print(Hidden_layer1)    # Tensor("Sigmoid:0", shape=(?, 30), dtype=float32)
# print(dropout_layers)   # Tensor("dropout/mul_1:0", shape=(?, 30), dtype=float32)
