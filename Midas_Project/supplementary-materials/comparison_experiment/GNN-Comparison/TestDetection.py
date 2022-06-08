#!/usr/bin/env python
# coding: utf-8

import sys
import os.path as osp
from random import random
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GraphConv, TopKPooling,GCNConv, GINEConv,TAGConv,MFConv,GATConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
import os
import json


device = 'cpu'
detection_data_test = torch.load('./data/datalist_detection_test.pt')


data_MD52label = np.load('./data/data_MD52label.npz', allow_pickle=True)
data_MD52label = data_MD52label['data_MD52label'][()]
md52filename = dict()

def loadnpz(hash_file):
    graph = np.load('./data/graph/' + hash_file + '.npz_graph.npz', allow_pickle=True)
    node_feature = graph['node_feature']
    edge_set = graph['edge_set']
    edge_attr = graph['edge_attr']
    d = Data()
    d.x = torch.tensor(node_feature)
    d.edge_attr = torch.tensor(edge_attr)
    d.edge_index = torch.tensor(edge_set).T
    d.additional = hash_file
    return d



class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.dec1 = torch.nn.Linear(100, 64)
        self.dec2 = torch.nn.Linear(64, 32)
        
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(32, 128),
            torch.nn.ReLU(True),
            torch.nn.Linear(128, 256)
        )
        self.gineconv1 = GINEConv(self.mlp)

        self.pool1 = TopKPooling(256, ratio=0.8)
        self.conv2 = GATConv(256, 64, heads=4)
        self.pool2 = TopKPooling(256, ratio=0.8)

        self.lin1 = torch.nn.Linear(512, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, 2)
        
    def forward(self, data):
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
        
        edge_attr = F.relu(self.dec1(edge_attr))
        edge_attr = F.relu(self.dec2(edge_attr))

        x = F.relu(self.gineconv1(x=x, edge_index=edge_index, edge_attr=edge_attr))
        x, edge_index, edge_attr1, batch, _, _ = self.pool1(x, edge_index, edge_attr, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, edge_attr2, batch, _, _ = self.pool2(x, edge_index, edge_attr1, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x



model = torch.load('./model/detection_model.pkl', map_location=torch.device('cpu'))


data_test_loader = DataLoader(detection_data_test, batch_size=128) 
data2md5 = dict()



def test(loader):
    for md5 in list(data_MD52label.keys()):
        data = loadnpz(md5)
        data.y = torch.tensor([0])
        data2md5[data.additional] = md5
    model.eval()
    label_list = []
    pred_list = []
    correct = 0
    for data in loader:
        data = data.to(device)
        pred = model(data).max(dim=1)[1]
        for i in range(len(pred)):
            if data[i].y != pred[i]:
                print(data2md5[data[i].additional])
                print(md52filename[data[i].additional])
        correct += pred.eq(data.y).sum().item()
        label_list_batch = data.y.to(device).detach().numpy().tolist()
        pred_list_batch = pred.to(device).detach().numpy().tolist()
        for label_item in label_list_batch:
            label_list.append(label_item)
        for pred_item in pred_list_batch:
            pred_list.append(pred_item)
    
    y_true = np.asarray(label_list)
    y_pred = np.asarray(pred_list)
    _val_confusion_matrix = confusion_matrix(y_true, y_pred)
    _val_acc = accuracy_score(y_true, y_pred)
    _val_precision = precision_score(y_true, y_pred)
    _val_recall = recall_score(y_true, y_pred)
    _val_f1 = f1_score(y_true, y_pred)
    return _val_confusion_matrix, _val_acc, _val_precision, _val_recall, _val_f1



behavior_lists = os.listdir("./original_behaviors/")
root = "./original_behaviors/"
for file in behavior_lists:
    if os.path.isfile(root+file):
        continue
    reader = open(root+file+"/task.json", "r")
    jsons = reader.read()
    reader.close()
    obj = json.loads(jsons)
    filename = obj["target"].split("/")[-1]
    pos = filename.find("_")
    filename = filename[pos+1:]
    md52filename[file] = filename

con, acc, precision, recall, f1 = test(data_test_loader)

print('Results Can Be Seen In The Following: ')
print('Test Acc: {:.5f}, Test Precision: {:.5f}, Test Recall: {:.5f}, Test F1: {:.5f}'.
          format(acc, precision, recall, f1))
print('FPR:', con[0,1]/(con[0,1]+con[0,0]))
print('FNR:', con[1,0]/(con[1,0]+con[1,1]))




