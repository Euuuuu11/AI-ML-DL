from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
import numpy as np  


datasets = load_diabetes()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(x,y,
             train_size=0.8,random_state=72)


# print(x)
# print(y)
# print(x.shape, y.shape) # (442, 10) (442, )

# print(datasets.feature_names)    # ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
# print(datasets.DESCR)   

#2. 모델구성
model = Sequential()
model.add(Dense(16, input_dim=10,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam',metrics=['mae'])

from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=400, mode='min', 
              verbose=1, restore_best_weights=True) 
model.fit(x_train, y_train, epochs=1000, validation_split=0.2,
                 batch_size=15, verbose=1, callbacks=[es])

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)


# 1. validation 사용 후
# loss :  2193.794921875
# r2스코어 :  0.6677688351801708

# 2. EarlyStopping 후
# loss :  2180.1416015625
# r2스코어 :  0.6698365357420499

# 3. activation 적용
# loss :  2159.809326171875
# r2스코어 :  0.6729156826833541

# 1,2,3 을 순차적으로 사용해보니, loss값이랑
# r2스코어 둘 다 좋게 개선되었다.