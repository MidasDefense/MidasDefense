#!/usr/bin/env python
# coding: utf-8
import numpy as np
import os

def ifNewNode(nodevec, nodevec_list):
    for n in nodevec_list:
        if all(nodevec == n):
            return False
    return True

def getNodeId2NodeVec(data):
    nid2nodevec = {0: -np.ones((32,), dtype=np.float32)}  # Node 0 is the end node of the graph
    for i in range(len(data)):
        nodevec = data[i, :32]
        if ifNewNode(nodevec, nid2nodevec.values()):
            nid2nodevec[len(nid2nodevec.keys())] = nodevec
    return nid2nodevec

def getEdgeSet(data, nid2nodevec):
    node_np = data[:, :32]
    item_list = []
    for i in range(len(node_np)):
        key = [k for k, v in nid2nodevec.items() if all(v == node_np[i])]
        item_list.append(key[0])
    edge_list = []
    for i in range(len(item_list)):
        if i == len(item_list) - 1:
            edge_list.append([item_list[i], 0])
        else:
            edge_list.append([item_list[i], item_list[i + 1]])
    return np.asarray(edge_list)

def getGraph(data):
    nid2nodevec = getNodeId2NodeVec(data)
    node_feature = np.asarray(list(nid2nodevec.values()))  # node feature
    edge_set = getEdgeSet(data, nid2nodevec)  # edge_set
    edge_attr = data[:, 32:]  # edge feature
    return node_feature, edge_set, edge_attr


callg_file_list = os.listdir('./data/sequence/')
num = 1
for graph_md5 in callg_file_list:
    print('{} / {}'.format(num, len(callg_file_list)))
    num += 1
    data = np.load('./data/sequence/' + graph_md5, allow_pickle=True)
    data = data['data']
    node_feature, edge_set, edge_attr = getGraph(data)
    np.savez('./data/graph/' + graph_md5 + '_graph.npz', node_feature=node_feature, edge_set=edge_set, edge_attr=edge_attr)
