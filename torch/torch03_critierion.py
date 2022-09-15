import numpy as np
import torch
print(torch.__version__)        # 1.12.1

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch:',torch.__version__,'use divece:',DEVICE)  # torch: 1.12.1 use divece: cuda

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

x = torch.FloatTensor(x).unsqueeze(1).to(DEVICE)  # (3,) -> (3,1) shape를 늘려주는것 번호는 위치
y = torch.FloatTensor(y).unsqueeze(-1).to(DEVICE)  # 

print(x.shape,y.shape)
# torch.Size([3]) torch.Size([3]) > torch.Size([3, 1]) torch.Size([3, 1])
print(x,y)

#2. 모델

# model = Sequential()
model =nn.Linear(1,1).to(DEVICE)  # (input(x),ouput(y))


#3. 컴파일,훈련
# model.compile(loss='mse',optimizer='sgd')
criterion = nn.MSELoss()        # criterion(표준) =loss  
optimizer = optim.SGD(model.parameters(),lr=0.1) # Sgd: 경사하강법
# optim.Adam(model.parameters(),lr=0.01)

#4. 훈련
def train(model,criterion,optimizer,x,y):
    model.train()           # 훈련모드 (생략가능) > 디폴트이기 때문에
    optimizer.zero_grad()   # 1손실함수의 기울기를 초기화 w.grad 값이 역전파할 때 누적이되기 때문에 0으로 만들어줘야한다.
    h = model(x)
    # loss = criterion(h,y)   # 갱신된 w 와 y를 비교한다.
    # loss = nn.MSELoss()(h,y)   # 갱신된 w 와 y를 비교한다.
    loss = F.mse_loss(h,y)   # 갱신된 w 와 y를 비교한다.

    loss.backward()         # 2역전파
    optimizer.step()        # 3가중치를 적용  1,2,3 무조건들어감 

    return loss.item()


epochs = 1000
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
results = model(torch.Tensor([[4]]).to(DEVICE))
print("4의 예측:",results.item())
    
'''
EPOCH = 100 //-1.# 총 epoch 수
learning_rate = 1e-5 #//-2. learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5) # //-3.  SGD optimizer
model.train() #//-4. train mode로 전환
for epoch in range(EPOCH):
    prediction = model(x) # //-5. 추론
    loss = F.mse_loss(prediction, y) # //-6. 손실(loss) 구하기
    optimizer.zero_grad() //-7. #gradient 초기화
    loss.backward() //-8. #자동미분
    optimizer.step() //-9. #weight 업데이트
    if epoch % 10 == 0:
    # //-10. 10번마다 로그 출력
      print('Epoch {:4d}/{} Cost: {:.6f}'.format(
          epoch, EPOCH, loss.item()
      ))
'''