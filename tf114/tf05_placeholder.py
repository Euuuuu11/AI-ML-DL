# input에만 관여
# 넣을 데이터를 정의할 떄 사용

import tensorflow as tf
import numpy as np
print(tf.__version__)
print(tf.executing_eagerly())   # True

#즉시 실행모드
tf.compat.v1.disable_eager_execution()  # 즉시 실행모드 꺼
print(tf.executing_eagerly())   # Flase

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

sess = tf.compat.v1.Session()

a = tf.compat.v1.placeholder(tf.float32)
b = tf.compat.v1.placeholder(tf.float32)

add_node = a + b

print(sess.run(add_node, feed_dict={a:3, b:4.5}))
print(sess.run(add_node, feed_dict={a:[1, 3], b:[2, 4]}))

add_and_triple = add_node * 3
print(add_and_triple)

print(sess.run(add_and_triple, feed_dict={a:3, b:4.5}))
# print(sess.run(add_node, feed_dict={a:[1, 3], b:[2, 4]}))

