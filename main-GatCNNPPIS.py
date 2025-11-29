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
import argparse as agp
import random
import os
import time
import torch.utils.data.sampler as sampler
from model_GatCNNPPIS import *
from data_generator_new import *
import pickle
import timeit
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score, precision_recall_curve, auc, accuracy_score, matthews_corrcoef
import pandas as pd

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as ddp
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


def stack_fn_cnn_ddp(batch):
    indexs, all_seq_features, labels = [], [], []
    for i in batch:
        index, seq, label = cnn_dataSet[i]
        indexs.append(index)
        all_seq_features.append(seq)
        labels.append(label)

    locals_len = 0
    proteins_len = 0
    N = len(labels)

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

    labels_new = np.zeros(N, dtype=np.long)
    i = 0
    for label in labels:
        labels_new[i] = label
        i += 1

    indexs_new = np.stack(indexs_new)
    proteins_new = np.stack(proteins_new)
    labels_new = np.stack(labels_new)

    return proteins_new, indexs_new, labels_new

def stack_fn_cnn(batch):
    indexs, all_seq_features, labels = [], [], []
    for index, seq, label in batch:
        indexs.append(index)
        all_seq_features.append(seq)
        labels.append(label)

    locals_len = 0
    proteins_len = 0
    N = len(labels)

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

    labels_new = np.zeros(N, dtype=np.long)
    i = 0
    for label in labels:
        labels_new[i] = label
        i += 1

    indexs_new = np.stack(indexs_new)
    proteins_new = np.stack(proteins_new)
    labels_new = np.stack(labels_new)

    return proteins_new, indexs_new, labels_new


def main(seed):
    init_seeds(seed)

    """Load preprocessed data."""

    with open('./data_cache/422_dset_list.pkl', "rb") as fp:
        all_list = pickle.load(fp)

    with open('./data_cache/training_list.pkl', "rb") as fp:
        train_dev_list = pickle.load(fp)

    with open('./data_cache/testing_list.pkl', "rb") as fp:
        test_list = pickle.load(fp)

    id_idxs = [item[1] for item in all_list]

    train_dev_id_idxs = [id_idxs[i] for i in train_dev_list]
    id_idxs_drop = list(set(train_dev_id_idxs))
    print(len(id_idxs_drop))  ##352
    # np.random.seed(1)
    np.random.shuffle(id_idxs_drop)
    train_id_idxs = id_idxs_drop[50:]
    dev_id_idxs = id_idxs_drop[:50]

    train_list = [i for i, x in enumerate(id_idxs) if x in train_id_idxs]
    
    # print(train_list)
    dev_list = [i for i, x in enumerate(id_idxs) if x in dev_id_idxs]
    # print(dev_list)

    if torch.cuda.is_available() and torch.cuda.device_count() >= 1:
        train_samples = torch.utils.data.distributed.DistributedSampler(train_list)
        dev_samples = torch.utils.data.distributed.DistributedSampler(dev_list)
    else:
        train_samples = sampler.SubsetRandomSampler(train_list)
        dev_samples = sampler.SubsetRandomSampler(dev_list)


    batch_size = 128
    if torch.cuda.is_available() and torch.cuda.device_count() >= 1:
        train_loader = torch.utils.data.DataLoader(train_list, batch_size=batch_size,
                                                sampler=train_samples,
                                                num_workers=0, collate_fn=stack_fn_cnn_ddp, drop_last=False)

        valid_loader = torch.utils.data.DataLoader(dev_list, batch_size=batch_size,
                                                sampler=dev_samples,
                                                num_workers=0, collate_fn=stack_fn_cnn_ddp, drop_last=False)
    else:
        train_loader = torch.utils.data.DataLoader(cnn_dataSet, batch_size=batch_size,
                                               sampler=train_samples,
                                               num_workers=0, collate_fn=stack_fn_cnn, drop_last=False)

        valid_loader = torch.utils.data.DataLoader(cnn_dataSet, batch_size=batch_size,
                                               sampler=dev_samples,
                                               num_workers=0, collate_fn=stack_fn_cnn, drop_last=False)
    

    """ create model, trainer and tester """

    protein_dim = 1024
    local_dim = 1024
    hid_dim = 64
    n_layers = 3
    n_heads = 8
    pf_dim = 256
    dropout = 0.1
    lr = 5e-4
    weight_decay = 1e-4
    decay_interval = 5
    lr_decay = 1.0
    iteration = 50
    kernel_size = 7

    encoder2 = Encoder2(protein_dim, hid_dim, n_layers, kernel_size, dropout, device)
    decoder2 = Decoder2(local_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer2, SelfAttention2,
                      PositionwiseFeedforward2, dropout, device)
    model = Predictor2(encoder2, decoder2, device)

    model.to(device)
    if torch.cuda.is_available() and torch.cuda.device_count() >= 1:
        model = ddp(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    trainer = Trainer2(model, lr, weight_decay)
    tester = Tester2(model)

    # Output files.
    os.makedirs(('./output/result'), exist_ok=True)
    os.makedirs(('./output/model'), exist_ok=True)
    file_AUCs = './output/result/output-GatCNNPPIS' + '.txt'
    file_model = './output/model/' + 'GatCNNPPIS'
    AUCs = ('Epoch\tTime1(sec)\tTime2(sec)\tLoss_train\tACC_dev\tAUC_dev\tRec_dev\tPre_dev\tF1_dev\tMCC_dev\tPRC_dev')
    with open(file_AUCs, 'w') as f:
        f.write(AUCs + '\n')

    # Start training.
    print('Training...')
    print(AUCs)

    max_PRC_dev = 0

    for epoch in range(1, iteration + 1):
        if torch.cuda.is_available() and torch.cuda.device_count() >= 1:
            train_loader.sampler.set_epoch(epoch)

        start = timeit.default_timer()
        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay

        loss_train = trainer.train(train_loader, device)

        end1 = timeit.default_timer()
        time1 = end1 - start


        correct_labels_valid, predicted_labels_valid, predicted_scores_valid = tester.test(valid_loader, device)
        if torch.cuda.is_available() and torch.cuda.device_count() >= 1:
            correct_labels_valid = torch.tensor(np.array(correct_labels_valid), dtype=torch.float64, device='cuda')
            predicted_labels_valid = torch.tensor(np.array(predicted_labels_valid), dtype=torch.float64, device='cuda')
            predicted_scores_valid = torch.tensor(np.array(predicted_scores_valid), dtype=torch.float64, device='cuda')

            correct_labels_valid_list_tmp = [torch.zeros_like(correct_labels_valid).cuda() for _ in
                                            range(torch.distributed.get_world_size())]
            dist.all_gather(correct_labels_valid_list_tmp, correct_labels_valid)
            correct_labels_valid_list = torch.cat(correct_labels_valid_list_tmp, dim=0).cpu()

            predicted_labels_valid_list_tmp = [torch.zeros_like(predicted_labels_valid).cuda() for _ in
                                            range(torch.distributed.get_world_size())]
            dist.all_gather(predicted_labels_valid_list_tmp, predicted_labels_valid)
            predicted_labels_valid_list = torch.cat(predicted_labels_valid_list_tmp, dim=0).cpu()

            predicted_scores_valid_list_tmp = [torch.zeros_like(predicted_scores_valid).cuda() for _ in
                                            range(torch.distributed.get_world_size())]
            dist.all_gather(predicted_scores_valid_list_tmp, predicted_scores_valid)
            predicted_scores_valid_list = torch.cat(predicted_scores_valid_list_tmp, dim=0).cpu()
        
            dist.barrier()  # synchronizes all processes
            
            ACC_dev, AUC_dev, Rec_dev, Pre_dev, F1_dev, MCC_dev, PRC_dev = metrics(correct_labels_valid_list,
                                                                                predicted_labels_valid_list,
                                                                                predicted_scores_valid_list)
        else: 
            ACC_dev, AUC_dev, Rec_dev, Pre_dev, F1_dev, MCC_dev, PRC_dev = metrics(correct_labels_valid,
                                                                    predicted_labels_valid,
                                                                    predicted_scores_valid)
        end2 = timeit.default_timer()
        time2 = end2 - end1
        AUCs = [epoch, time1, time2, loss_train, ACC_dev, AUC_dev, Rec_dev, Pre_dev, F1_dev, MCC_dev, PRC_dev]
        tester.save_AUCs(AUCs, file_AUCs)

        if PRC_dev > max_PRC_dev:
            last_improve = epoch
            print('last_improve: %s' % last_improve)
            if torch.cuda.is_available() and torch.cuda.device_count() >= 1:
                if dist.get_rank() == 0:
                    tester.save_model(model, file_model)
            else:
                torch.save(model.state_dict(), file_model)
            max_PRC_dev = PRC_dev
        print('\t'.join(map(str, AUCs)))
        if epoch - last_improve >= 10:
            print('errly stopping at epoch: %s' % epoch)
            break



if __name__ == "__main__":
    """CPU or GPU"""
    if torch.cuda.is_available():
        if torch.cuda.device_count() >= 1:

            # 1
            dist.init_process_group(backend='nccl')

            # 2
            parser = argparse.ArgumentParser()
            parser.add_argument("--local_rank", default=0, type=int,
                                help='node rank for distributed training')
            args = parser.parse_args()

            # 3
            local_rank = args.local_rank

            # 4
            torch.cuda.set_device(local_rank)

            # 5
            device = torch.device("cuda", local_rank)

        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')


    """Load preprocessed data."""

    all_encode_file = './data_cache/422_encode_data.pkl'
    all_label_file = './data_cache/422_label.pkl'
    all_list_file = './data_cache/422_dset_list.pkl'

    window_size = 0
    cnn_dataSet = dataSet_cnn(window_size, all_encode_file, all_label_file, all_list_file)

    SEED = 1
    main(SEED)


