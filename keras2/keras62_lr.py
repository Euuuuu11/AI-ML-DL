
x = 10
y = 10
w = 0.5
lr = 0.001
epochs = 5000000

for i in range(epochs):
    predict = x * w
    loss = (predict - y) ** 2
    
    print('Loss : ', round(loss, 4), '\tPredict : ', round(predict, 4))
    
    up_predict = x * (w + lr)
    up_loss = (y - up_predict) ** 2
    
    down_predict = x * (w - lr)
    down_loss = (y - down_predict) ** 2
    
    if(up_loss > down_loss):
        w = w - lr
    else:
        w = w + lr
 
    
    
    
    
    
    
    