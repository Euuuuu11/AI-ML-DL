# 1 357
# 0 212

# 라벨 0을 112개 삭제해서 재구성

# smote 넣어서 만들기
# 넣은거 안넣은거 비교
# smote 넣은거 안넣은거 비교
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE    # pip install imblearn
import sklearn as sk
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

x = data.data
y = data.target
# print(pd.Series(y).value_counts())
# 1    357
# 0    212  --> 112개 삭제
#print(x.shape, y.shape)  # (569, 30) (569,)

index_list = np.where(y==0) # y에서 0이 들어있는 인덱스 위치가 담긴 리스트
print(len(index_list[0])) # 212

del_index_list = index_list[0][100:]
print(len(del_index_list))    # 112

new_x = np.delete(x,del_index_list,axis=0) # del_index_list
new_y = np.delete(y,del_index_list)
        
print(pd.Series(new_y).value_counts())
# 1    357
# 0    100


x_train, x_test, y_train, y_test = train_test_split(new_x, new_y , random_state=123, shuffle=True, 
                                                    train_size=0.75)

print(pd.Series(y_train).value_counts())


#2. 모델
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

#3. 훈련
model.fit(x_train,y_train)

#4. 평가,예측
y_predict = model.predict(x_test)

from sklearn.metrics import accuracy_score, f1_score    

score = model.score(x_test, y_test)
# print('model.score : ', score)
print('acc_score : ', accuracy_score(y_test,y_predict))
print('f1_score(macro) : ', f1_score(y_test,y_predict, average='macro'))
# print('f1_score(micro) : ', f1_score(y_test,y_predict, average='micro'))

# 기본결과
# acc_score :  0.9777777777777777
# f1_score(macro) :  0.9797235023041475

# 데이터 축소 후(2라벨을 25개 줄인 후)
# acc_score :  0.9428571428571428
# f1_score(macro) :  0.8596176821983273
print('============================ SMOTE 적용 후 ============================')
smote = SMOTE(random_state=123,k_neighbors=1)
x_train, y_train = smote.fit_resample(x_train,y_train)

# print(pd.Series(y_train).value_counts())
# 0    53
# 1    53
# 2    53

#2. 모델, #3. 훈련
model = RandomForestClassifier()
model.fit(x_train, y_train)

#4. 평가,예측
y_predict = model.predict(x_test)

from sklearn.metrics import accuracy_score, f1_score    

score = model.score(x_test, y_test)
# print('model.score : ', score)
print('acc_score : ', accuracy_score(y_test,y_predict))
print('f1_score(macro) : ', f1_score(y_test,y_predict, average='macro'))
# print('f1_score(micro) : ', f1_score(y_test,y_predict, average='micro'))

# acc_score :  0.9652173913043478
# f1_score(macro) :  0.9515993265993266
# ============================ SMOTE 적용 후 ============================
# acc_score :  0.9652173913043478
# f1_score(macro) :  0.9538893344025661