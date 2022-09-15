import numpy as np
import torch
print(torch.__version__)        # 1.12.1

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 사이킷런 스케일링 단점 : 행렬의 곱셈이 불가능하다.

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch:',torch.__version__,'use divece:',DEVICE)  # torch: 1.12.1 use divece: cuda

#1. 데이터
x = torch.FloatTensor([range(10),range(21,31),range(201,211)]).to(DEVICE)

y = torch.FloatTensor([[1,2,3,4,5,6,7,8,9,10],
             [1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9],
             [9,8,7,6,5,4,3,2,1,0]]).to(DEVICE)
x_test = torch.FloatTensor([9, 30, 210]).to(DEVICE)
print(x.shape,y.shape, x_test.shape)  # torch.Size([3, 1]) torch.Size([3, 1]) torch.Size([1, 1])

#reshape x
x = x.T
y = y.T

# x_test = torch.FloatTensor(x_test).unsqueeze(1).to(DEVICE)

# x_test = (x_test - x.mean()) / x.std()  # 평균을 빼고 표준편차로 나눈다., std = 표준편차
# x = (x - x.mean()) / x.std()  # 스탠다드 스케일링
print(x, y, x_test)

print(x.shape,y.shape, x_test.shape)  # torch.Size([3, 1]) torch.Size([3, 1]) torch.Size([1, 1])
# torch.Size([3]) torch.Size([3]) > torch.Size([3, 1]) torch.Size([3, 1])
#2. 모델

# model =nn.Linear(1,5).to(DEVICE)  # (1,1) : 1개의 입력을 받아서 1개의 출력을 내보낸다.
# model =nn.Linear(5,3).to(DEVICE)  #
# model =nn.Linear(3,4).to(DEVICE)  #
# model =nn.Linear(4,2).to(DEVICE)  #
# model =nn.Linear(2,1).to(DEVICE)  # nn : 신경망을 만들어주는 모듈

model = nn.Sequential(
    nn.Linear(3,50),
    nn.Linear(50,40),
    nn.Linear(40,30),
    nn.ReLU(),
    nn.Linear(30,20),
    nn.Linear(20,3)).to(DEVICE)




#3. 컴파일,훈련
# model.compile(loss='mse',optimizer='sgd')
criterion = nn.MSELoss()        # criterion(표준) =loss  
# optimizer = optim.SGD(model.parameters(),lr=0.00005)  # model.parameters() : 가중치를 가져온다.
optimizer = optim.Adam(model.parameters(),lr=0.01)

#4. 훈련
def train(model,criterion,optimizer,x,y):
    model.train()           # 훈련모드 (생략가능) > 디폴트이기 때문에
    optimizer.zero_grad()   # 1손실함수의 기울기를 초기화 w.grad 값이 역전파할 때 누적이되기 때문에 0으로 만들어줘야한다.
    h = model(x)
    loss = criterion(h,y)   # 갱신된 w 와 y를 비교한다.
    # loss = nn.MSELoss()(h,y)   # 갱신된 w 와 y를 비교한다.
    # loss = F.mse_loss(h,y)   # 갱신된 w 와 y를 비교한다.

    loss.backward()         # 2 역전파 : 손실함수를 미분해서 기울기를 구한다.
    optimizer.step()        # 3 가중치를 적용  1,2,3 무조건들어감 

    return loss.item()


epochs = 500
for epoch in range(1,epochs+1):
    loss = train(model, criterion,optimizer,x,y )
    print('epoch :{},loss:{}'.format(epoch,loss))
    
# 평가,예측
# loss = model.evaluate(x,y)  
def evaluate(model, criterion,x,y ) :   # 평가에서는 optimizer가 필요없다.
    model.eval()                  #평가모드 > 디폴트가 train이기 때문에 평가모드 지정해야한다. 
    
    with torch.no_grad():         # 역전파 하지 않고 순전파만 사용하겠다. w 갱신하지 않겠다는 의미 평가하는 구간이기 때문에.
        y_predict = model(x)      # x_test
        results = criterion(y_predict,y)
    return results.item()

loss2 = evaluate(model,criterion,x,y)
print('최종loss:',loss2)
# y_predict = model.predict([4])
results = model(x_test)
# results = torch.Tensor.cpu(results)
print('예측값:',results)  # detach() : 미분을 하지 않겠다.  cpu() : gpu에서 cpu로 옮겨준다.  numpy() : numpy로 바꿔준다.
print('예측값:',results.detach().cpu().numpy())  # detach() : 미분을 하지 않겠다.  cpu() : gpu에서 cpu로 옮겨준다.  numpy() : numpy로 바꿔준다.