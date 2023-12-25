# -*- coding: utf-8 -*-
"""
@Time:Created on 2022/11/12 10:00
@author: Minjie Mou
@Filename: main.py
@Software: PyCharm
"""
import torch
import numpy as np
import sys
import random
import os
import time
import torch.utils.data.sampler as sampler
from model_TransformerPPIS import *
from predict_data_generator import *
from model_GatCNNPPIS import *
from predict_data_generator_new import *
import pickle
import timeit
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score, roc_curve, precision_recall_curve, auc, accuracy_score, matthews_corrcoef
import pandas as pd
import matplotlib.pyplot as plt

import argparse


def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def metrics(correct_labels, predicted_labels, predicted_scores):
    ACC = accuracy_score(correct_labels, predicted_labels)
    AUC = roc_auc_score(correct_labels, predicted_scores)
    CM = confusion_matrix(correct_labels, predicted_labels)
    TN = CM[0][0]
    FP = CM[0][1]
    FN = CM[1][0]
    TP = CM[1][1]
    Rec = TP / (TP + FN)
    Pre = TP / (TP + FP)
    F1 = 2 * Pre * Rec / (Pre + Rec)
    MCC = matthews_corrcoef(correct_labels, predicted_labels)
    precision, recall, _ = precision_recall_curve(correct_labels, predicted_scores)
    PRC = auc(recall, precision)
    return ACC, AUC, Rec, Pre, F1, MCC, PRC

def stack_fn(batch):
    local_features, all_seq_features = [], []
    for local, seq in batch:
        local_features.append(local)
        all_seq_features.append(seq)
    
    locals_len = 0
    proteins_len = 0
    N = len(local_features)
    local_num = []
    protein_num = []

    local_dim = 1024
    protein_dim = 1024
    for local in local_features:
        local_num.append(local.shape[0])
        if local.shape[0] >= locals_len:
            locals_len = local.shape[0]
    for protein in all_seq_features:
        protein_num.append(protein.shape[0])
        if protein.shape[0] >= proteins_len:
            proteins_len = protein.shape[0]
        
    locals_new = np.zeros((N, locals_len, local_dim))
    i = 0
    for local in local_features:
        # print(local.shape)
        a_len = local.shape[0]
        # print(a_len)
        locals_new[i, :a_len, :] = local
        i += 1

    proteins_new = np.zeros((N, proteins_len, protein_dim))
    i = 0
    for protein in all_seq_features:
        # print(protein.shape)
        a_len = protein.shape[0]
        # print(a_len)
        proteins_new[i, :a_len, :] = protein
        i += 1
        
    locals_new = np.stack(locals_new)
    proteins_new = np.stack(proteins_new)

    return locals_new, proteins_new, local_num, protein_num


def stack_fn_cnn(batch):
    indexs, all_seq_features = [], []
    for index, seq in batch:
        indexs.append(index)
        all_seq_features.append(seq)

    locals_len = 0
    proteins_len = 0
    N = len(indexs)

    local_dim = 1024
    protein_dim = 1024

    for protein in all_seq_features:
        if protein.shape[0] >= proteins_len:
            proteins_len = protein.shape[0]

    indexs_new = np.zeros(N, dtype=np.long)
    i = 0
    for index in indexs:
        indexs_new[i] = index
        i += 1

    proteins_new = np.zeros((N, proteins_len, protein_dim))
    i = 0
    for protein in all_seq_features:
        # print(protein.shape)
        a_len = protein.shape[0]
        # print(a_len)
        proteins_new[i, :a_len, :] = protein
        i += 1

    indexs_new = np.stack(indexs_new)
    proteins_new = np.stack(proteins_new)

    return proteins_new, indexs_new
    

def main(seed):
    init_seeds(seed)

    """Load preprocessed data."""

    all_encode_file = './data_cache/Test_70_ProtBERT_data.pkl'

    test_file = './data_cache/Test_70.tsv'

    all_encodes = []
    with open(all_encode_file,"rb") as fp_enc:
        temp_enc  = pickle.load(fp_enc)
    all_encodes.extend(temp_enc)

    test_df = pd.read_csv(test_file, sep='\t')


    """ create model and tester """

    protein_dim1 = 1024
    local_dim1 = 1024
    hid_dim = 64
    n_layers = 3
    n_heads = 8
    pf_dim = 256
    dropout = 0.1
    lr = 5e-4
    weight_decay = 1e-4
    decay_interval = 5
    lr_decay = 1.0

    kernel_size = 7
    
    encoder1 = Encoder(protein_dim1, hid_dim, n_layers, kernel_size, dropout, device)
    decoder1 = Decoder(local_dim1, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)
    model1 = Predictor(encoder1, decoder1, device)
    model1.load_state_dict(torch.load("./output/model/trained-TransformerPPIS", map_location=torch.device('cpu')))

    model1.to(device)

    tester1 = Predictor_test(model1)


    encoder2 = Encoder2(protein_dim1, hid_dim, n_layers, kernel_size, dropout, device)
    decoder2 = Decoder2(local_dim1, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer2, SelfAttention2, PositionwiseFeedforward2, dropout, device)
    model2 = Predictor2(encoder2, decoder2, device)
    model2.load_state_dict(torch.load("./output/model/trained-GatCNNPPIS", map_location=torch.device('cpu')))

    model2.to(device)

    tester2 = Predictor_test2(model2)

    """Output files."""

    output_file = './output/result/predict-Test70-EnsemPPIS' + '.txt'
    metrics_file = './output/result/metrics-Test70-EnsemPPIS' + '.txt'

    start = timeit.default_timer()

    f = open(output_file, 'w')
    m = open(metrics_file, 'w')
    AUCs = ('name\tlength\tACC\tAUC\tRec\tPre\tF1\tMCC\tPRC')
    m.write(AUCs + '\n')
    correct_labels_test_all = []
    predicted_labels_list_all = []
    predicted_scores_list_all = []

    for index in range(len(all_encodes)):
        print(index)
        protein = all_encodes[index]

        ii,protein_id,seq_length = [], [], []

        name = test_df['name'].iloc[index]
        seq = test_df['sequence'].iloc[index]
        correct_labels_test = list(map(int, test_df['label'].iloc[index]))
        correct_labels_test_all.extend(correct_labels_test)
        print(len(correct_labels_test_all))
        
        length = len(protein)

        for i in range(len(seq)):
            ii.append(i)
            protein_id.append(name)
            seq_length.append(length)

        tulpe = list(zip(ii,protein_id,seq_length))

        window_size = 0
        all_dataSet = dataSet(window_size, protein, tulpe)
        cnn_dataSet = dataSet_cnn(window_size, protein, tulpe)

        test_samples = sampler.SequentialSampler(tulpe)

        batch_size = 256

        test_loader1 = torch.utils.data.DataLoader(all_dataSet, batch_size=batch_size,
                                                sampler=test_samples, 
                                                num_workers=0, collate_fn=stack_fn, drop_last=False)
        
        test_loader2 = torch.utils.data.DataLoader(cnn_dataSet, batch_size=batch_size,
                                                sampler=test_samples, 
                                                num_workers=0, collate_fn=stack_fn_cnn, drop_last=False)

        predicted_labels_test1, predicted_scores_test1 = tester1.test(test_loader1, device)
        predicted_labels_test2, predicted_scores_test2 = tester2.test(test_loader2, device)
        predicted_labels_test3 = []
        predicted_scores_test3 = []
        for i in range(len(predicted_labels_test1)):
            predicted_score = (predicted_scores_test1[i] + predicted_scores_test2[i])/2
            predicted_scores_test3.append(predicted_score)
            if (predicted_scores_test1[i] + predicted_scores_test2[i]) >= 1.2:
                predicted_labels_test3.append(int(1))
            else:
                predicted_labels_test3.append(int(0))
        predicted_labels_list_all.extend(predicted_labels_test3)
        predicted_scores_list_all.extend(predicted_scores_test3)
        print(len(predicted_labels_list_all))

        ACC, AUC, Rec, Pre, F1, MCC, PRC = metrics(correct_labels_test, predicted_labels_test3, predicted_scores_test3)
        all = [length, ACC, AUC, Rec, Pre, F1, MCC, PRC]
        m.write(name + '\t' + '\t'.join(map(str, all)) + '\n')

        predicted_labels_list1 = list(map(lambda x:str(x),predicted_labels_test1))
        predicted_labels_list2 = list(map(lambda x:str(x),predicted_labels_test2))
        predicted_labels_list3 = list(map(lambda x:str(x),predicted_labels_test3))

        f.write('>' + name + '\n')
        f.write(str(seq) + '\n')
        f.writelines(predicted_labels_list1)
        f.write('\n')
        f.writelines(predicted_labels_list2)
        f.write('\n')
        f.writelines(predicted_labels_list3)
        f.write('\n')

    f.close()

    ACC, AUC, Rec, Pre, F1, MCC, PRC = metrics(correct_labels_test_all, predicted_labels_list_all, predicted_scores_list_all)
    print("ACC: ", ACC)
    print("Pre: ", Pre)
    print("Rec: ", Rec)
    print("F1: ", F1)
    print("AUC: ", AUC)
    print("PRC: ", PRC)
    print("MCC: ", MCC)


    end = timeit.default_timer()
    time = end - start


if __name__ == "__main__":
    """CPU or GPU"""
    if torch.cuda.is_available() :
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        device = torch.device('cuda:0')

        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    SEED = 1
    main(SEED)

