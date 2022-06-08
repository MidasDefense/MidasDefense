#!/usr/bin/env python
# coding: utf-8


import numpy as np
import os
import torch
from torch_geometric.data import Data, DataLoader, Dataset



label = np.load('./data/label.npz', allow_pickle=True)
label = label['label'][()]

dataset_split_md5_list = np.load('./dataset_split_md5_list.npz', allow_pickle=True)
detection_train_md5 = dataset_split_md5_list['detection_train_md5']
detection_test_md5 = dataset_split_md5_list['detection_test_md5']




def loadnpz(hash_file):
    graph = np.load('./data/graph/' + hash_file + '.npz_graph.npz', allow_pickle=True)
    node_feature = graph['node_feature']
    edge_set = graph['edge_set']
    edge_attr = graph['edge_attr']
    d = Data()
    d.x = torch.tensor(node_feature)
    d.edge_attr = torch.tensor(edge_attr)
    d.edge_index = torch.tensor(edge_set).T
    return d

datalist_detection_train = []
datalist_detection_test = []
num = 1
for graph_md5 in list(label.keys()):
    print('{} / {}'.format(num, len(label)))
    num += 1
    if label[graph_md5] == 'Benign':
        data = loadnpz(graph_md5)
        data.y = torch.tensor([0])
        data.additional = graph_md5
        if graph_md5 in detection_train_md5:
            datalist_detection_train.append(data)
        else:
            datalist_detection_test.append(data)
    else:
        data = loadnpz(graph_md5)
        data.y = torch.tensor([1])
        data.additional = graph_md5
        if graph_md5 in detection_train_md5:
            datalist_detection_train.append(data)
        else:
            datalist_detection_test.append(data)
    
'''
Samples Are Verified by VirusTotal, including the label and its discovery time.
'''
torch.save(datalist_detection_train, './data/datalist_detection_train.pt')
torch.save(datalist_detection_test, './data/datalist_detection_test.pt')
