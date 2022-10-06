# [과제]
# 3가지 원핫인코딩 방식을 비교할것
# 
#1. pandas의 get_dummies
# 결측값이 사라져서 수가 줄어듦
#2. tensorflow의 to_categorical
# categorical은 앞에 0부터 시작 0이 없으면 0을 생성시킴.
#3. sklearn의 oneHotEncoder
# from sklearn.preprocessing import OneHotEncoder
# ohe = OneHotEncoder(sparse=False)  #if sparse = True면 metrics로 출력, False면 array로 출력
# y = ohe.fit_transform(y.reshape(-1,1))  #1부터 시작 ~ -1즉 배열 끝까지 출력
# 미세한 차이를 정리하시오

 