import tensorflow as tf
tf.compat.v1.set_random_seed(123)

변수 = tf.compat.v1.Variable(tf.random_normal([1]), name= 'weight')
print(변수)

#1. 초기화
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
aaa = sess.run(변수)
print('aaa : ', aaa)    # aaa :  [-1.5080816]
sess.close()

#2. 초기화
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
bbb = 변수.eval(session= sess)  # 이 과정을 거쳐야
print('bbb : ', bbb)    # bbb :  [-1.5080816]   # 변수로 
sess.close()

#3. 초기화
sess = tf.compat.v1.InteractiveSession()    # InteractiveSession는 eval 안에 파라미터를 지정하지 않아도 된다.
sess.run(tf.compat.v1.global_variables_initializer())
ccc = 변수.eval()
print('ccc : ', ccc)    # ccc :  [-1.5080816]
sess.close()





 