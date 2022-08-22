import tensorflow as tf
print(tf.__version__)           # 1.14.0
print(tf.executing_eagerly())   # False

# 즉시 실행 모드 (False는 버전 1점대, True는 버전 2점대)
# 즉시 실행모드 : 2점대 즉시 실행
tf.compat.v1.disable_eager_execution() 

print(tf.executing_eagerly())   # False

hello = tf.constant('Hello World')
sess = tf.compat.v1.Session()
print(sess.run(hello))  # 'Hello World'






