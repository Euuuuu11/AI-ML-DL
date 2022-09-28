from torchvision.datasets import *
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision.transforms as tr

USE_CUDA = torch.cuda.is_available()                   
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')
print(torch.__version__, '사용DVICE:', DEVICE)


#1. 데이터
path = './_data/torch_data/'

transf = tr.Compose([tr.Resize(15),tr.ToTensor()])

# train_dataset = MNIST(path, train=True,download=True,transform=transf)
# test_dataset = MNIST(path, train=False,download=True,transform=transf)

train_dataset = CIFAR100(path, train=True,download=False)
test_dataset = CIFAR100(path, train=False,download=False)

# print(train_dataset[0][0].shape)

x_train,y_train = train_dataset.data/255, train_dataset.targets
x_test,y_test = test_dataset.data/255, test_dataset.targets

x_train = torch.FloatTensor(x_train).to(DEVICE)
y_train = torch.LongTensor(y_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_test = torch.LongTensor(y_test).to(DEVICE)

print(x_train.shape,x_test.shape)          # torch.Size([60000, 28, 28]) torch.Size([10000, 28, 28])
print(x_test.shape,x_test.shape)           # torch.Size([10000, 28, 28]) torch.Size([10000, 28, 28])

# print(np.min(x_train),np.max(x_train))      # 0.0 1.0


x_train, x_test = x_train.reshape(50000,3,32,32),x_test.reshape(10000,3,32,32)      # torch reshape 방법
# x_train, x_test = x_train.unsqueeze(1), x_test.unsqueeze(1)  
print(x_train.shape,x_test.shape)       # torch.Size([50000, 32, 32, 3]) torch.Size([10000, 32, 32, 3])

train_dset = TensorDataset(x_train,y_train)
test_dset = TensorDataset(x_test,y_test)

train_loader = DataLoader(train_dset, batch_size=128,shuffle=True)
test_loader = DataLoader(test_dset, batch_size=128,shuffle=False)


#2.모델
# class DNN(nn.Module):
#     def __init__(self, num_features):
#         super().__init__()

#         self.hidden_layer1 = nn.Sequential(nn.Linear(num_features,128),
#                                            nn.ReLU(),
#                                            nn.Dropout(0.1))
#         self.hidden_layer2 = nn.Sequential(nn.Linear(128,64),
#                                            nn.ReLU(),
#                                            nn.Dropout(0.1))
#         self.hidden_layer3 = nn.Sequential(nn.Linear(64,32),
#                                            nn.ReLU(),
#                                            nn.Dropout(0.1),)    
#         self.hidden_layer4 = nn.Sequential(nn.Linear(32,16),
#                                            nn.ReLU(),
#                                            nn.Dropout(0.1))
#         self.output_layer = nn.Linear(16,10)
class CNN(nn.Module):
    def __init__(self, num_features):
        super(CNN,self).__init__()

        self.hidden_layer1 = nn.Sequential(nn.Conv2d(num_features,128, kernel_size=(3,3),stride=1),
                                           nn.ReLU(),
                                           nn.MaxPool2d(kernel_size=(2,2)),
                                           nn.Dropout(0.1))
        self.hidden_layer2 = nn.Sequential(nn.Conv2d(128,32, kernel_size=(3,3)),
                                            nn.ReLU(),
                                            nn.MaxPool2d(kernel_size=(2,2)),
                                            nn.Dropout(0.1))
    
        self.hidden_layer3 = nn.Linear(32*6*6,32)
        

        self.output_layer = nn.Linear(in_features =32,out_features = 100)    
            
    def forward(self,x):
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = x.view(x.shape[0],-1)     # flatten/ -1 : 32*5*5         
        x = self.hidden_layer3(x)
        x = self.output_layer(x)
        return x 

model = CNN(3).to(DEVICE)
# from torchsummary import summary 
# summary(model,(1,32,32))   


#3.컴파일훈련

criterion = nn.CrossEntropyLoss().to(DEVICE)
optimizer = optim.Adam(model.parameters(),lr=1e-4)  #0.0001

def train(model,criterion,optimizer,loader):
    
    epoch_loss = 0
    epoch_acc = 0
    
    for x_batch, y_batch in loader :
        x_batch,y_batch = x_batch.to(DEVICE),y_batch.to(DEVICE)
                
        optimizer.zero_grad()
        
        h = model(x_batch)
        
        loss = criterion(h,y_batch)   
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        y_predict = torch.argmax(h,1)
        acc = (y_predict == y_batch).float().mean()
        
        epoch_acc += acc.item()
        
    return epoch_loss / len(loader), epoch_acc / len(loader)
        
# hist  = model.fit (x_trian, y_train)      # hist 에는 loss와 acc가 들어가
# 엄밀하게 얘기하면 hist라고 하기는그렇고 , loss와 acc를 반환해준다고함.

def evaluate (model,criterion,loader):
    model.eval()                # eval 모드에서는 dropout의 기능이 하지 않도록 설정이 되어있다.
    
    epoch_loss = 0
    epoch_acc = 0
    
    with torch.no_grad():
        for x_batch,y_batch in loader:
            x_batch,y_batch = x_batch.to(DEVICE),y_batch.to(DEVICE)
            
            h = model(x_batch)
            
            loss = criterion(h, y_batch)
            
            epoch_loss += loss.item()
            y_predict = torch.argmax(h,1)
            acc = (y_predict == y_batch).float().mean()
        
            epoch_acc += acc.item()
        
        return epoch_loss / len(loader), epoch_acc / len(loader)



epochs =17
for epoch in range(1,epochs +2):
    
    loss,acc =train(model,criterion,optimizer, train_loader)
    val_loss,val_acc = evaluate(model,criterion,test_loader)
    
    print('epochs:{},loss{:.4f},acc:{:.3f},val_loss:{:.4f},val_acc{:.4f}'.format(epoch, loss, acc, val_loss,val_acc))        
     
        
                                                                                      
                                           
