from time import time
    
import torch
from transformers import BertModel, BertTokenizer
import re
import os
from tqdm.auto import tqdm
import numpy as np
import gzip
import pickle
import pandas as pd

def generate_protbert_features(file):
    path = './'
    t0=time()

    modelFilePath = path+'pytorch_model.bin'

    configFilePath = path+'config.json'

    vocabFilePath = path+'vocab.txt'
        
    tokenizer = BertTokenizer(vocabFilePath, do_lower_case=False )
    model = BertModel.from_pretrained(path)
    device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    model = model.eval()

    sequences = []
    df = pd.read_csv(file, sep=',')
    sequences =df['sequence'].values.tolist()
        
    sequences_Example = [' '.join(list(seq)) for seq in sequences]
    sequences_Example = [re.sub(r"[-UZOB]", "X", sequence) for sequence in sequences_Example]

    all_protein_features = []

    for i, seq in enumerate(sequences_Example):
        print(i)
        ids = tokenizer.batch_encode_plus([seq], add_special_tokens=True, pad_to_max_length=False)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)
        with torch.no_grad():
            embedding = model(input_ids=input_ids,attention_mask=attention_mask)[0]
        embedding = embedding.cpu().numpy()
        features = []
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][1:seq_len-1]
            features.append(seq_emd)
            
            print(features.__len__())
        # print(all_protein_sequences['all_protein_complex_pdb_ids'][i])
    #     print(features)
        all_protein_features += features

    pickle.dump(all_protein_features, open('../data_cache/422_encode_data.pkl','wb')) 
    ##["dset186","dset164","dset72"]_encode_data.pkl

    print('Total time spent for ProtBERT:',time()-t0)

if __name__ == "__main__":

    file =  '../data_cache/422_name_seq_label.csv' 

    generate_protbert_features(file)
