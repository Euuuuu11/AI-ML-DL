import pandas as pd
import numpy as np
import glob
import joblib


path = 'D:\study_data\_data\green/'
all_input_list = sorted(glob.glob(path + 'train_input/*.csv'))
all_target_list = sorted(glob.glob(path + 'train_target/*.csv'))
test_input_list = sorted(glob.glob(path + 'test_input/*.csv'))
test_target_list = sorted(glob.glob(path + 'test_target/*.csv'))

train_input_list = all_input_list[:50]
train_target_list = all_target_list[:50]

val_input_list = all_input_list[50:]
val_target_list = all_target_list[50:]






# print(all_input_list)
print(val_input_list)
print(len(val_input_list))  # 8

def aaa(input_paths, target_paths): #, infer_mode):
    input_paths = input_paths
    target_paths = target_paths
    # self.infer_mode = infer_mode
   
    data_list = []
    label_list = []
    print('시작...')
    # for input_path, target_path in tqdm(zip(input_paths, target_paths)):
    for input_path, target_path in zip(input_paths, target_paths):
        input_df = pd.read_csv(input_path)
        target_df = pd.read_csv(target_path)
       
        input_df = input_df.drop(columns=['시간'])
        input_df = input_df.fillna(0)
       
        input_length = int(len(input_df)/1440)
        target_length = int(len(target_df))
        print(input_length, target_length)
       
        for idx in range(target_length):
            time_series = input_df[1440*idx:1440*(idx+1)].values
            # self.data_list.append(torch.Tensor(time_series))
            data_list.append(time_series)
        for label in target_df["rate"]:
            label_list.append(label)
    return np.array(data_list), np.array(label_list)
    print('끗.')

train_data, label_data = aaa(train_input_list, train_target_list) #, False)
val_train_data, val_label_data = aaa(val_input_list, val_target_list) #, False)
test_input_data, test_target_data = aaa(test_input_list, test_target_list) #, False)

train_data = np.save('D:\study_data\_save\_npy/train_data1.npy', train_data)
label_data = np.save('D:\study_data\_save\_npy/label_data2.npy',label_data)
val_data = np.save('D:\study_data\_save\_npy/val_data1.npy',val_train_data)
val_target = np.save('D:\study_data\_save\_npy/val_target1.npy',val_label_data)
test_data = np.save('D:\study_data\_save\_npy/test_data1.npy',test_input_data)
test_target = np.save('D:\study_data\_save\_npy/test_target1.npy',test_target_data)

exit()
print(train_data[0])
print(len(train_data), len(label_data)) # 1607 1607
print(len(train_data[0]))   # 1440
print(label_data)   # 1440
print(train_data.shape, label_data.shape)   # (1607, 1440, 37) (1607,)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten, Dropout, MaxPooling1D, GRU

model = Sequential()
model.add(LSTM(50,input_shape=(1440,37)))
# model.add(GRU(50, activation='relu'))
# model.add(GRU(50))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
model.summary()

# 3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(train_data, label_data, epochs=1, batch_size=100, validation_data=(val_train_data, val_label_data))

y_summit = model.predict(test_data)

path2 = 'D:\study_data\_data\_csv\dacon_grow\\test_target/' # ".은 현재 폴더"
targetlist = ['TEST_01.csv','TEST_02.csv','TEST_03.csv','TEST_04.csv','TEST_05.csv','TEST_06.csv']
# [29, 35, 26, 32, 37, 36]
empty_list = []
for i in targetlist:
    test_target2 = pd.read_csv(path2+i)
    empty_list.append(test_target2)
    
empty_list[0]['rate'] = y_summit[:29]
empty_list[0].to_csv(path2+'TEST_01.csv')
empty_list[1]['rate'] = y_summit[29:29+35]
empty_list[1].to_csv(path2+'TEST_02.csv')
empty_list[2]['rate'] = y_summit[29+35:29+35+26]
empty_list[2].to_csv(path2+'TEST_03.csv')
empty_list[3]['rate'] = y_summit[29+35+26:29+35+26+32]
empty_list[3].to_csv(path2+'TEST_04.csv')
empty_list[4]['rate'] = y_summit[29+35+26+32:29+35+26+32+37]
empty_list[4].to_csv(path2+'TEST_05.csv')
empty_list[5]['rate'] = y_summit[29+35+26+32+37:]
empty_list[5].to_csv(path2+'TEST_06.csv')
# submission = submission.fillna(submission.mean())
# submission = submission.astype(int)

import os
import zipfile
filelist = ['TEST_01.csv','TEST_02.csv','TEST_03.csv','TEST_04.csv','TEST_05.csv', 'TEST_06.csv']
os.chdir("D:\study_data\_data\_csv\dacon_grow/test_target")
with zipfile.ZipFile("D:\study_data\_data\_csv\dacon_grow/sample_submission.zip", 'w') as my_zip:
    for i in filelist:
        my_zip.write(i)
    my_zip.close()
print('Done')
print('R2 :', r2)
print('RMSE :', rmse)
end_time = time.time()-start_time
print('걸린 시간:', end_time)


exit()

#4. 평가, 예측

loss, mae = model.evaluate(val_train_data, val_label_data, batch_size=1)
print('loss, mae : ', loss, mae)

y_pred = model.predict(test_input_data)
print(y_pred)
print(y_pred.shape)  # (11520, 1)




test = []
test_split_list = [29, 35, 26, 32, 37, 36]

for [index, value] in enumerate(test_split_list):
    new_list = y_pred[:value]
    csv = pd.read_csv(path + 'test_target/' + 'TEST_' + '%02d' %(index+1) + '.csv')
    csv['rate'] = new_list
    csv.to_csv(path + 'TEST_' + '%02d' %(index+1) + '.csv', index = True)
    y_pred = y_pred[value:]
    
    
print(test)   # (11520,)
    



# joblib.save(y_pred, 'C:\study\_data\dacon_chung/y_pred.pkl')