#-*- encoding:utf8 -*-

import os
import time
import pickle
import torch
import torch.utils.data.sampler as sampler
import numpy as np
from torch.utils import data
import pandas as pd


class dataSet_cnn(data.Dataset):
    def __init__(self, window_size, encode_data=None, protein_list=None):
        super(dataSet_cnn,self).__init__()

        self.all_encodes = encode_data
        self.protein_list = protein_list

        self.window_size = window_size



    def __getitem__(self,index):
        
        ii,protein_id,seq_length = self.protein_list[index]
        window_size = self.window_size
        

        win_start = ii - window_size
        win_end = ii + window_size
        seq_length = int(seq_length)
        label_idx = (win_start+win_end)//2

        all_seq_features = []
        all_seq_features = self.all_encodes

        ## local features
        local_features = []
        while win_start < 0:
            data = []
            acid_embed = [0 for i in range(len(self.all_encodes[0]))]
            data.extend(acid_embed)
            local_features.append(data)
            win_start += 1

        valid_end = min(win_end,seq_length-1)
        while win_start <= valid_end:
            data = []
            acid_embed = self.all_encodes[win_start]
            data.extend(acid_embed)
            local_features.append(data)
            win_start += 1

        while win_start <= win_end:
            data = []
            acid_embed = [0 for i in range(len(self.all_encodes[0]))]
            data.extend(acid_embed)
            local_features.append(data)
            win_start += 1


        index = ii
        index = np.array(index,dtype=np.float32)
        
        all_seq_features = np.stack(all_seq_features)
        all_seq_features = all_seq_features.astype(float)


        return index, all_seq_features

    def __len__(self):
    
        return len(self.protein_list)



