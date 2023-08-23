#-*- encoding:utf8 -*-

import os
import time
import pickle
import torch as t
import numpy as np
from pandas import DataFrame
import pandas as pd



# Test_355
f1 = open('./Datasets/DELPHI_task/Test_355.fa','r')

byt1 = f1.readlines()

data1 = []
for line in byt1:
    line = line.strip()
    data1.append(line)

dict1 = {'data': data1}
df1 = DataFrame(dict1)

df1['data'].replace(regex=True,inplace=True,to_replace=r'>',value=r'')

name, sequence, label = [], [], []
i = 0
while i < 355:
    name.append(df1['data'].iloc[3*i])
    sequence.append(df1['data'].iloc[3*i+1])
    label.append(df1['data'].iloc[3*i+2])
    i += 1

dict_new = {'name': name,
        'sequence': sequence,
        'label': label}
new = DataFrame(dict_new)
print(new.shape[0])
new.to_csv('Test_355_name_seq_label.csv', index = False)

# Test_355 label
sequences = []
df = pd.read_csv('Test_355_name_seq_label.csv', sep=',')
sequences = df['label'].values.tolist()
print(sequences[0][0])
pickle.dump(sequences, open('Test_315_label.pkl','wb')) 

# Test_355 list
df = pd.read_csv('Test_355_name_seq_label.csv', sep=',')
sequences = []
sequences = df['sequence'].values.tolist()
names = df['name'].values.tolist()
count,id_idx,ii,dset,protein_id,seq_length = [], [], [], [], [], []

all_count = 0
idx = 0
for index in range(len(sequences)):
    seq = sequences[index]
    name = names[index]
    length = len(seq)
    for i in range(len(seq)):
        count.append(all_count)
        id_idx.append(idx)
        ii.append(i)
        dset.append('Test_355')
        protein_id.append(name)
        seq_length.append(length)
        all_count = all_count + 1
    idx = idx + 1

tulpe = list(zip(count,id_idx,ii,dset,protein_id,seq_length))
print(len(tulpe))
pickle.dump(tulpe, open('Test_355_dset_list.pkl','wb')) 



