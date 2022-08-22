import tensorflow as tf
sess = tf.compat.v1.Session()

x = tf.Variable([2], dtype=tf.float32)
y = tf.Variable([3], dtype=tf.float32)

# 초기값을 넣을 상태를 만들어줌.
init = tf.compat.v1.global_variables_initializer()
sess.run(init)  # 실행을 시켜줘야함

print(sess.run(x+y))

