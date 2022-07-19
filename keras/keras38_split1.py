from posixpath import split
import numpy as np
from sklearn import datasets

dataset = np.array([1,2,3,4,5,6,7,8,9,10])
                                        # split_xy1 함수, dataset, time_step 변수 설정
def split_xy1(dataset, time_steps):     # dataset은 자르고자하는 데이터셋, time_stes는 몇 개의 컬럼으로 자를건지
    x, y = list(), list()               # 리턴해줄 x,y를 리스트로 정의
    for i in range(len(dataset)) :      
        end_number = i  + time_steps    # dataset 개수만큼 for문 돌림, 마지막 숫자가 몇인지를 정의해줌
        if end_number > len(dataset) -1:
            break                       # 마지막 숫자가 dataset의 전체길이에서 1개 뺀 값보다 크면 for문 정지
        tmp_x, tmp_y = dataset[i:end_number], dataset[end_number] # i가 0으로 과정했을 때 tmp_x는 dataset[0:4]이므로 1,2,3,4, tmp_y는 dataset[4]이므로 5가 된다.
        x.append(tmp_x)                                           # for문을 통해 마지막 숫자가 10이 넘지 않을 때까지 반복하여 리스트에 append로 붙게된다.
        y.append(tmp_y)
    return np.array(x), np.array(y)     # for문이  모두 끝나면 이 함수는 x,y값을 반환한다.
    
x,y = split_xy1(dataset,3)
print(x, "\n", y)    # 줄바꿈        
# [[1 2 3]
#  [2 3 4]
#  [3 4 5]
#  [4 5 6]
#  [5 6 7]
#  [6 7 8]
#  [7 8 9]] 
#  [ 4  5  6  7  8  9 10]     
    
    
def split_xy3(dataset, time_steps, y_column) : 
    x, y = list(), list()
    for i in range(len(dataset)) :
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column -1
        
        if y_end_number > len(dataset) :
            break
        tmp_x = dataset[i:x_end_number, :-1]
        tmp_y = dataset[x_end_number-1:y_end_number, -1]
        x.append(tmp_x)
        y.append(tmp_y)
        return np.array(X), np.array(y)
    x,y = split_xy3(dataset, 3, 2)
    print(x,"\n", y)
    print(x.shape)
    print(y.shape)
        