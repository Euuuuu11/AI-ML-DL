import tensorflow as tf
print("tf버전 :", tf.__version__)

hello = tf.constant('hello world')  # Tensor("Const:0", shape=(), dtype=string)
# hello = tf.compat.v1.constant('hello world')
print(hello)

# sess = tf.Session()
sess = tf.compat.v1.Session()
print(sess.run(hello))  # b'hello world'
# tensorflow는 출력할때 반드시 session을 사용해야한다.(sess.run으로 사용. tensorflow2부터는 사라졌음.)






