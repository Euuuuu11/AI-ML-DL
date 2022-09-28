from calendar import EPOCH
from sklearn.datasets import load_wine
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용DEVICE : ', DEVICE)

#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets['target']

x = torch.FloatTensor(x)
y = torch.LongTensor(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.7, shuffle=True, random_state=123, stratify=y)

x_train = torch.FloatTensor(x_train)
# y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
# y_train = torch.FloatTensor(y_train).to(DEVICE)
y_train = torch.LongTensor(y_train).to(DEVICE)

x_test = torch.FloatTensor(x_test)
# y_test = torch.FloatTensor(y_test).unsqueeze(-1).to(DEVICE)
# y_test = torch.FloatTensor(y_test).to(DEVICE)
y_test = torch.LongTensor(y_test).to(DEVICE)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)

print(x_train.size())   # torch.Size([124, 13])
print(x_train.shape)    # torch.Size([124, 13])

############################# 시작 #############################
from torch.utils.data import TensorDataset, DataLoader
train_set = TensorDataset(x_train, y_train) # x와 y를 합친다
test_set = TensorDataset(x_test, y_test) # x와 y를 합친다

print(train_set)    # <torch.utils.data.dataset.TensorDataset object at 0x0000019493120CA0>
print('='*60)
print(train_set[0])
print('='*60)
print(train_set[0][0])
print('='*60)
print(len(train_set))   # 398
print('='*60)

train_loader = DataLoader(train_set, batch_size=40, shuffle=True)
test_loader = DataLoader(test_set, batch_size=40, shuffle=False)

# exit()
#2. 모델
# model = nn.Sequential(
#     nn.Linear(13, 64),
#     nn.ReLU(),
#     nn.Linear(64, 32),
#     nn.ReLU(),
#     nn.Linear(32, 16),
#     nn.ReLU(),
#     nn.Linear(16, 3),
#     # nn.Sigmoid()
# ).to(DEVICE)

class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        # super().__init__()
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_dim,64)
        self.linear2 = nn.Linear(64, 16)        
        self.linear3 = nn.Linear(16, 8)
        self.linear4 = nn.Linear(8, output_dim)        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_size):
        x = self.linear1(input_size)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.sigmoid(x)
        return x 

model = Model(13,3).to(DEVICE)

#3. 컴파일, 훈련
criterion = nn.CrossEntropyLoss()       

optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, loader):
    # model.train()
    total_loss = 0
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        hypothesis = model(x_batch)
        loss = criterion(hypothesis, y_batch)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


EPOCHS = 100
for epoch in range(1, EPOCH+1):
    loss = train(model, criterion, optimizer, train_loader)
    print('epoch : {}, loss : {:.8f}'.format(epoch, loss))
    
#4. 평가, 예측
print('============= 평가, 예측 =============')    
def evaluate(model, criterion,loader):
    model. eval()
    total_loss = 0
    
    for x_batch, y_batch in loader:
        with torch.no_grad():
            hypothesis = model(x_test)
            loss = criterion(hypothesis,y_test)
            total_loss += loss.item()
            
    return total_loss

loss = evaluate(model, criterion, test_loader)
print('loss : ', loss)

# y_predict = model(x_test)
# print(y_predict[:10])

y_predict = (model(x_test) >= 0.5).float()
print(y_predict[:10])
y_predict = torch.argmax(y_predict, axis=1)
score = (y_predict == y_test).float().mean()
print('accuracy :  {:.4f}'.format(score))      

from sklearn.metrics import accuracy_score
# score = accuracy_score(y_test, y_predict) # error
# print('accuray_score : ', score)

score = accuracy_score(y_test.cpu().numpy(), y_predict.cpu().numpy())
print('accuray_score : ', score)

# accuracy :  0.9630
# accuray_score :  0.9629629629629629