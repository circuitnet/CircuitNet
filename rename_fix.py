import os
import pandas as pd
import csv
    

def changeFilenameSubstring():
    ref_dic = {}

    data = pd.read_csv('name_reference.csv',sep=',',header=None,usecols=[1,2])
    old = pd.read_csv('./congestion_v4/congestion_train.csv',sep=',',header=None,usecols=[0,1])
    for idx,row in data.iterrows():
        ref_dic[row[1]] = str(idx+1) + '-' + row[2]

    for row in old.iterrows():
        print(row[1][0])
        if os.path.basename(row[1][0])[:-4] in ref_dic:
            new_row = [os.path.dirname(row[1][0]) +'/'+ ref_dic[os.path.basename(row[1][0])[:-4]] + '.npy', os.path.dirname(row[1][1]) +'/'+ ref_dic[os.path.basename(row[1][1])[:-4]] + '.npy']
            if not os.path.exists(os.path.dirname(row[1][0]) +'/'+ ref_dic[os.path.basename(row[1][0])[:-4]] + '.npy'):
                print('skip')
                continue
        elif 'f0_z1' in row[1][0]:
            temp1 = row[1][0].replace('f0_z1', 'f1_z0')
            temp2 = row[1][1].replace('f0_z1', 'f1_z0')
            if os.path.basename(temp1)[:-4] not in ref_dic:
                continue
            new_row = [os.path.dirname(temp1) +'/'+ ref_dic[os.path.basename(temp1)[:-4]] + '.npy', os.path.dirname(temp2) +'/'+ ref_dic[os.path.basename(temp2)[:-4]] + '.npy']
            if not os.path.exists(os.path.dirname(temp1) +'/'+ ref_dic[os.path.basename(temp1)[:-4]] + '.npy'):
                print('skip')
                continue
        elif 'f1_z0' in row[1][0]:
            temp1 = row[1][0].replace('f1_z0', 'f0_z1')
            temp2 = row[1][1].replace('f1_z0', 'f0_z1')
            if os.path.basename(temp1)[:-4] not in ref_dic:
                continue
            new_row = [os.path.dirname(temp1) +'/'+ ref_dic[os.path.basename(temp1)[:-4]] + '.npy', os.path.dirname(temp2) +'/'+ ref_dic[os.path.basename(temp2)[:-4]] + '.npy']
            if not os.path.exists(os.path.dirname(temp1) +'/'+ ref_dic[os.path.basename(temp1)[:-4]] + '.npy'):
                print('skip')
                continue
        with open('./congestion_train.csv', 'a') as f:
            f_csv = csv.writer(f, delimiter=',')
            f_csv.writerow(new_row)


changeFilenameSubstring()

