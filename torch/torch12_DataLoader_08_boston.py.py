from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_diabetes, load_boston
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# import to_categorical
from sklearn.preprocessing import OneHotEncoder

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

#1. 데이터
dataset = load_boston()
x = dataset.data
y = dataset['target']

x = torch.FloatTensor(x)
y = torch.FloatTensor(y).unsqueeze(1)




from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,
                                                    train_size=0.8,shuffle=True,
                                                    random_state=66)


from sklearn.preprocessing import StandardScaler, MinMaxScaler
# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.FloatTensor(y_train).to(DEVICE)
y_test = torch.FloatTensor(y_test).to(DEVICE)

############################# 시작 #############################
from torch.utils.data import TensorDataset, DataLoader
train_set = TensorDataset(x_train, y_train) # x와 y를 합친다
test_set = TensorDataset(x_test, y_test) # x와 y를 합친다

print(train_set)    # <torch.utils.data.dataset.TensorDataset object at 0x0000019493120CA0>
print('='*80)
print(train_set[0])
print('='*80)
print(train_set[0][0])
print('='*80)
print(len(train_set))   # 398
print('='*80)

train_loader = DataLoader(train_set, batch_size=40, shuffle=True)
test_loader = DataLoader(test_set, batch_size=40, shuffle=True)

#2. 모델
# model = nn.Sequential(
#     nn.Linear(13,100),
#     nn.ReLU(),
#     nn.Linear(100,200),
#     nn.ReLU(),
#     nn.Linear(200,150),
#     nn.ReLU(),
#     nn.Linear(150,50),
#     nn.ReLU(),
#     nn.Linear(50,1)).to(DEVICE)

class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        # super().__init__()
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_dim,100)
        self.linear2 = nn.Linear(100, 200)        
        self.linear3 = nn.Linear(200, 150)
        self.linear4 = nn.Linear(150, 50)
        self.linear5 = nn.Linear(50, output_dim)        
        self.relu = nn.ReLU()
        
    def forward(self, input_size):
        x = self.linear1(input_size)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.relu(x)
        x = self.linear5(x)
        return x 

model = Model(13,1).to(DEVICE)

#3. 컴파일,훈련

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

def train(model,criterion,optimizer,loader):
    model.train()
    total_loss = 0
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        prediction = model(x_batch)
        loss = criterion(prediction,y_batch)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(loader)

def r2_score(y_test,y_pred):
    u = ((y_test-y_pred)**2).sum()
    v = ((y_test-y_test.mean())**2).sum()
    return 1-u/v

epochs = 1000
for epoch in range(1, epochs+1):
    loss = train(model,criterion,optimizer,train_loader)
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Loss: {:.6f} r2_score: {:.4f}%'.format(
            epoch,epochs,loss,r2_score(y_test,model(x_test))))
        
#4. 평가,예측

def evaluate(model,criterion,loader):
    model.eval()
    total_loss = 0
    
    for x_batch, y_batch in loader:
        with torch.no_grad():
            prediction = model(x_test)
            loss = criterion(prediction,y_test)
            total_loss += loss.item()
            
    return loss.item()
            
            
loss = evaluate(model,criterion,test_loader)

print('Loss: {:.6f}'.format(loss))
print('r2_score: {:.4f}%'.format(r2_score(y_test,model(x_test))))


# Loss: 15.357096
# r2_score: 0.8163%