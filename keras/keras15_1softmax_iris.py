import numpy as np
from pandas import Categorical
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from tensorflow.python.keras.models import Sequential   
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
import tensorflow as tf

# import tensorflow as tf
# tf.random.set_seed(15)

#1. 데이터
datasets = load_iris()
# print(datasets.DESCR) 
# print(datasets.feature_names) # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'] 
x = datasets['data']
y = datasets.target
# print(x)
# print(y)
# print(x.shape, y.shape) # (150, 4) (150, )
print("y의 라벨값 : ", np.unique(y)) # y의 라벨값 : [0 1 2]

# ValueError: in user code:
#     ValueError: Shapes (1, 1) and (1, 3) are incompatible
# y.shape를 (150,1) -> (150,3) 으로 바꿔줘야 해결이 된다.
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

# print(y)
# print(y.shape) # (150, 3)


x_train, x_test, y_train, y_test = train_test_split(x,y,
             train_size=0.8, shuffle=True, random_state=66)
# print(y_test)
# print(y_train)


#2. 모델구성
model = Sequential()
model.add(Dense(128, input_dim=4,activation='relu'))
model.add(Dense(64))
model.add(Dense(32,activation='relu'))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(3, activation='softmax')) # y의 분류하는 개수와 노드의 개수 동일

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy']) # 다중분류에는 loss = 'categorical_crossentropy'만 사용 

from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=400, mode='min', 
              verbose=1, restore_best_weights=True) 

model.fit(x_train, y_train,
          epochs=500, batch_size=100,validation_split=0.2,
          verbose=1, callbacks=[es])


#4. 평가, 예측
# loss, acc = model.evaluate(x_test, y_test)
# print("loss : ", loss) 
# print("acc : ", acc) # 이 3줄이랑 밑 3줄이랑 같다.

result = model.evaluate(x_test,y_test)
print("loss : ", result[0])
print("accuracy : ", result[1])

# print("=================y_test[:5]==================") 
# print(y_test[:5])
# print("===================y_pred===================") 
# print(y_pred)
print("============================================") 

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
# print(y_predict)
y_test = np.argmax(y_test, axis=1)
# print(y_test)

acc = accuracy_score(y_test, y_predict)
print('acc스코어 : ', acc)

























