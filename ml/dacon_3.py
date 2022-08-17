import pandas as pd

path  = 'D:/study_data/_data/dacon3/' \
    
train_set = pd.read_csv(path + 'train.csv', 
                        index_col=0) 


test_set = pd.read_csv(path + 'test.csv', 
                       index_col=0)


