
import numpy as np
import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_dense_adj
from scipy import sparse as sp

from utils.quat import normalize_adj

"""Convert a scipy sparse matrix to a torch sparse tensor."""
def sparse_mx_to_torch_sparse_tensor(sparse_mx, device='cpu'):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape).to(device)

""" quaternion preprocess for feature vectors """
def quaternion_preprocess_features(features, device='cpu'):
    """Row-normalize feature matrix"""
    features = features.cpu()
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    # print(features.shape)
    # features = features.todense()
    features = np.tile(features, 4) # A + Ai + Aj + Ak
    return torch.from_numpy(features).to(device)



class ToQuaternions(BaseTransform):
    r"""Converts a graph to quaternions.
    Args:
        num_features (int): The number of features.
    """
    def __init__(self):
        super().__init__()

    def __call__(self, data):
        data.x = quaternion_preprocess_features(data.x)
        adj_t = to_dense_adj(data.edge_index.cpu()).squeeze().numpy()
        adj_t = normalize_adj(adj_t + sp.eye(adj_t.shape[0])).tocoo()
        adj_t = sparse_mx_to_torch_sparse_tensor(adj_t, device=data.x.device)
        data.adj_t = adj_t 
        return data