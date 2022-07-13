import numpy as np

a = np.array(range(1, 13))
size = 6

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)
    
bbb = split_x(a, size)
print(bbb)
print(bbb.shape)    

x = bbb[:, :-1]
y = bbb[:, -1]
print(x,y)
print(x.shape, y.shape) 
    