import os
from functools import partial
import torch
import numpy as np
import scipy.sparse as sp
from torch_geometric.transforms import Compose, ToSparseTensor, RandomNodeSplit, RandomLinkSplit, NormalizeFeatures, ToDevice
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.utils import to_dense_adj, add_self_loops

from transforms import ToQuaternions
from utils.quat import normalize_adj
from tabulate import tabulate

supported_datasets = {
    'cora': partial(Planetoid, root='/datasets', name='Cora'),
    'citeseer': partial(Planetoid, root='/datasets', name='CiteSeer'),
    'pubmed': partial(Planetoid, root='/datasets', name='PubMed'),
    'mutag': partial(TUDataset, root='/datasets', name='MUTAG'),
    'proteins': partial(TUDataset, root='/datasets', name='PROTEINS'),
    'enzymes': partial(TUDataset, root='/datasets', name='ENZYMES'),
    'collab': partial(TUDataset, root='/datasets', name='COLLAB'),
}

supported_tasks = ['node', 'graph', 'link']

# quaternions
"""Convert a scipy sparse matrix to a torch sparse tensor."""
def sparse_mx_to_torch_sparse_tensor(sparse_mx, device='cuda'):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape).to(device)

""" quaternion preprocess for feature vectors """
def quaternion_preprocess_features(features, device='cuda'):
    """Row-normalize feature matrix"""
    features = features.cpu()
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    features = np.tile(features, 4) # A + Ai + Aj + Ak
    return torch.from_numpy(features).to(device)

def create_masks(num_samples, val_ratio, test_ratio):
    train_ratio = 1.0 - (test_ratio + val_ratio)
    

    # Shuffle the indices
    indices = torch.arange(num_samples)
    indices = torch.randperm(num_samples)

    # number of samples for each set
    num_train = int(train_ratio * num_samples)
    num_test = int(test_ratio * num_samples)
    num_val = int(val_ratio * num_samples)
    
    # print(num_train, num_test, num_val)

    # masks
    train_mask = torch.zeros(num_samples, dtype=bool)
    test_mask = torch.zeros(num_samples, dtype=bool)
    val_mask = torch.zeros(num_samples, dtype=bool)

    # indices to masks
    train_mask[indices[:num_train]] = True
    test_mask[indices[num_train:num_train + num_test]] = True
    val_mask[indices[num_train + num_test:num_train + num_test + num_val]] = True

    return train_mask, test_mask, val_mask

def print_dataset_info(data):
    if isinstance(data, list):  # For TUDatasets with multiple graphs
        dataset_info = [
            #["Dataset Name", data[0].name],
            ["Number of Graphs", len(data)],
            ["Average Number of Nodes", sum(d.num_nodes for d in data) / len(data)],
            ["Average Number of Features", sum(d.num_features for d in data) / len(data)],
            # Add more statistics as needed
        ]
        print(tabulate(dataset_info, headers=["Attribute", "Value"], tablefmt="fancy_grid"))
    else:  # For Planetoid datasets or single graph TUDatasets
        num_classes = data.num_mod_classes  
        dataset_info = [
            ["Dataset Name", data.name],
            ["Number of Nodes", data.num_nodes],
            ["Number of Features", data.num_features],
            ["Number of Classes", data.num_mod_classes]
        ]
        print(tabulate(dataset_info, headers=["Attribute", "Value"], tablefmt="fancy_grid"))

def load_dataset(
        dataset:        dict(help='Name of the dataset', option='-d', choices=supported_datasets) = 'cora',
        data_dir:       dict(help='Directory to save the dataset') = './datasets',
        val_ratio:      dict(help='Validation ratio') = 0.25,
        test_ratio:     dict(help='Test ratio') = 0.25,
        quat:           dict(help='Use quaternion representation') = True,
        task:           dict(help='Task to perform', choices=supported_tasks) = 'node',
        device:         dict(help='Device to run the training', choices=['cpu', 'cuda']) = 'cuda'
    ):
    
    if task == 'node':
        transform = Compose([
            NormalizeFeatures(),
            ToDevice(device),
            RandomNodeSplit(split='train_rest', num_val=val_ratio, num_test=test_ratio)
        ])
        data = supported_datasets[dataset](root=os.path.join(data_dir, dataset), transform=transform)[0]
        data = preprocess_batch(data)
        data.name = dataset
            
    elif task == 'graph':
        transform = Compose([
            NormalizeFeatures(),
            ToDevice(device)
        ])
        data = supported_datasets[dataset](root=os.path.join(data_dir, dataset), transform=transform)
        data = preprocess_batch(data)
        data.num_nodes = len(data.x)
        data.name = dataset
        data.train_mask, data.test_mask, data.val_mask = create_masks(len(data), val_ratio=val_ratio, test_ratio=test_ratio)
        
    elif task == 'link':
        transform = Compose([
            NormalizeFeatures(),
            RandomLinkSplit(num_val=val_ratio, num_test=test_ratio, is_undirected=True,
                      add_negative_train_samples=False, neg_sampling_ratio = 1.0),
            ToDevice(device),
        ])
        data = supported_datasets[dataset](root=os.path.join(data_dir, dataset), transform=transform)
        data = preprocess_batch(data)
        data.train_mask, data.test_mask, data.val_mask = create_masks(len(data), val_ratio=val_ratio, test_ratio=test_ratio)
        data.name = dataset

    return data  


"""Preprocessing has been tweaked to prevent 4x features scaling for quaternions.""" 
def preprocess_batch(data):
    data = preprocess_single_batch(data)
    return data

def preprocess_single_batch(data):
    num_extra_classes = 0

    num_features = data.x.shape[1]
    if num_features % 4 != 0:
        num_extra_features = 4 - (num_features % 4)
        avg_feature = torch.mean(data.x, dim=1, keepdim=True)
        while data.x.shape[1] % 4 != 0:
            data.x = torch.cat([data.x, avg_feature], dim=1)

    if hasattr(data, 'num_classes'):  # Planetoid datasets
        num_classes = data.num_classes
        if num_classes % 4 != 0:
            num_extra_classes = 4 - (num_classes % 4)
            data.num_mod_classes = num_classes + num_extra_classes  # New attribute to store extra classes

    elif hasattr(data, 'y'):  # TUDatasets
        num_classes = int(data.y.max().item() + 1)
        if num_classes % 4 != 0:
            num_extra_classes = 4 - (num_classes % 4)
            data.num_mod_classes = num_classes + num_extra_classes  # New attribute to store extra classes

    data.x = data.x.to(data.edge_index.device)

    # Adjoint Matrix
    adj_t = to_dense_adj(data.edge_index.cpu()).squeeze().numpy()
    adj_t = normalize_adj(adj_t + sp.eye(adj_t.shape[0])).tocoo()
    adj_t = sparse_mx_to_torch_sparse_tensor(adj_t, device=data.x.device)
    data.adj_t = adj_t

    return data

if __name__ == '__main__':
    data = load_dataset(dataset='cora', task='link', device='cuda', quat=True)
    print(data[0])
