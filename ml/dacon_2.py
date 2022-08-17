import numpy as np
import pandas as pd                               
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error,accuracy_score
from tqdm import tqdm_notebook
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
#1. 데이터
path = 'D:\study_data\_data/dacon2/'
train = pd.read_csv(path + 'train.csv',                 
                        index_col=0)                       

test = pd.read_csv(path + 'test.csv',                                   
                       index_col=0)

sample_submission = pd.read_csv(path + 'sample_submission.csv')


print(train.describe())  # DurationOfPitch, MonthlyIncome
print("=============================상관계수 히트 맵==============")
print(train.corr())                    # 상관관계를 확인.  
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set(font_scale=0.3)
sns.heatmap(data=train.corr(),square=True, annot=True, cbar=True) 
plt.show()



# 결측치를 처리하는 함수를 작성.
def handle_na(data):
    temp = data.copy()
    for col, dtype in temp.dtypes.items():
        if dtype == 'object':
            # 문자형 칼럼의 경우 'Unknown'
            value = 'Unknown'
        elif dtype == int or dtype == float:
            # 수치형 칼럼의 경우 0
            value = 0
        temp.loc[:,col] = temp[col].fillna(value)
    return temp

train_nona = handle_na(train)

# 결측치 처리가 잘 되었는지 확인해 줍니다.
train_nona.isna().sum()

print(train_nona.isna().sum())
object_columns = train_nona.columns[train_nona.dtypes == 'object']
print('object 칼럼 : ', list(object_columns))

# 해당 칼럼만 보아서 봅시다
train_nona[object_columns]

print(train_nona.shape)
print(test.shape)


# LabelEncoder를 준비해줍니다.
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

# LabelEcoder는 학습하는 과정을 필요로 합니다.
encoder.fit(train_nona['TypeofContact'])

#학습된 encoder를 사용하여 문자형 변수를 숫자로 변환해줍니다.
encoder.transform(train_nona['TypeofContact'])
print(train_nona['TypeofContact'])

train_enc = train_nona.copy()

# 모든 문자형 변수에 대해 encoder를 적용합니다.
for o_col in object_columns:
    encoder = LabelEncoder()
    encoder.fit(train_enc[o_col])
    train_enc[o_col] = encoder.transform(train_enc[o_col])

# 결과를 확인합니다.
print(train_enc)
# 결측치 처리
test = handle_na(test)

# 문자형 변수 전처리
for o_col in object_columns:
    encoder = LabelEncoder()
    
    # test 데이터를 이용해 encoder를 학습하는 것은 Data Leakage 입니다! 조심!
    encoder.fit(train_nona[o_col])
    
    # test 데이터는 오로지 transform 에서만 사용되어야 합니다.
    test[o_col] = encoder.transform(test[o_col])

# 결과를 확인
print(test)


# 모델 선언
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.linear_model import LogisticRegression
model = RandomForestClassifier()

# 분석할 의미가 없는 칼럼을 제거합니다.
# 상관계수 그래프를 통해 연관성이 적은것과 - 인것을 빼준다.
train = train_enc.drop(columns=['TypeofContact','NumberOfChildrenVisiting','NumberOfPersonVisiting','OwnCar', 'MonthlyIncome'])  
test = test.drop(columns=['TypeofContact','NumberOfChildrenVisiting','NumberOfPersonVisiting','OwnCar', 'MonthlyIncome'])


# 학습에 사용할 정보와 예측하고자 하는 정보를 분리합니다.
x_train = train.drop(columns=['ProdTaken'])
y_train = train[['ProdTaken']]
print(x_train.info())
print(y_train.info())


# 모델 학습
model.fit(x_train,y_train)

prediction = model.predict(test)
print('----------------------예측된 데이터의 상위 10개의 값 확인--------------------\n')

print(prediction[:10])
# print(model.score(x_train, y_train))
# 예측된 값을 정답파일과 병합
sample_submission['ProdTaken'] = prediction

# 정답파일 데이터프레임 확인
print(sample_submission)

sample_submission.to_csv(path+'sample_submission.csv',index = False)

exit()


drop_cols = [ 'Age', 'NumberOfTrips' ,'MonthlyIncome' ,'TypeofContact','Occupation','ProductPitched','MaritalStatus','Passport']

train_set.drop(drop_cols, axis = 1, inplace =True)
test_set.drop(drop_cols, axis = 1, inplace =True)

submission = pd.read_csv(path + 'sample_submission.csv',#예측에서 쓸거야!!
                       index_col=0)

print(train_set.describe()) 
print(train_set.columns.values)

# ['Age' 'TypeofContact' 'CityTier' 'DurationOfPitch' 'Occupation' 'Gender'
#  'NumberOfPersonVisiting' 'NumberOfFollowups' 'ProductPitched'
#  'PreferredPropertyStar' 'MaritalStatus' 'NumberOfTrips' 'Passport'
#  'PitchSatisfactionScore' 'OwnCar' 'NumberOfChildrenVisiting'
#  'Designation' 'MonthlyIncome' 'ProdTaken']
# Age / NumberOfTrips /MonthlyIncome 열 삭제 
# TypeofContact / DurationOfPitch /NumberOfFollowups /PreferredPropertyStar /NumberOfTrips  /NumberOfChildrenVisiting /MonthlyIncome  nan값 해결
###### 결측치 처리 1.제거##### dropna 사용
train_set.astype(float)
test_set.astype(float)

print(train_set.isnull().sum()) #각 컬럼당 결측치의 합계
train_set = train_set.fillna(train_set.median())
print(train_set.isnull().sum())
print(train_set.shape)
test_set = test_set.fillna(test_set.median())
print(train_set.shape)  # (1955, 19)
print(test_set.shape)   # (2933, 18)



x = train_set.drop(['ProdTaken'],axis=1)  
print(x.shape) #(1955, 10)
y = train_set['ProdTaken']
print(y.shape) #(1955,)

y = np.array(y).reshape(-1, 1)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.25,
                                                    random_state=58525
                                                    )

print(x_train.shape) # (712, 7)
print(y_train.shape) # (712, 1)
print(x_test.shape) # (179, 7)
print(y_test.shape) # (179, 1)


#2. 모델구성
model = Sequential()
model.add(Dense(10, activation='selu', input_dim=7))
model.add(Dense(100, activation='selu'))
model.add(Dense(80, activation='selu'))
model.add(Dense(15, activation='selu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1, 
                              restore_best_weights=True)


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=34, verbose=1, 
          validation_split=0.2, callbacks=[earlyStopping])


#4. 평가, 예측

loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', acc)

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis= 1)
y_test = np.argmax(y_test, axis= 1)

acc1 = accuracy_score(y_test, y_predict) 
print('acc1 : ', acc1) 