import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def t_o(n):
   matrix = torch.zeros((n, n))
   for i in range(n):
      matrix[i, (i*4)%n + (i*4)//n] = 1
      
   return matrix

def make_quaternion_mul(kernel, device=device):
    """The constructed 'hamilton' W is a modified version of the quaternion representation,
       thus doing tf.matmul(Input,W) is equivalent to W * Inputs."""
    dim = kernel.size(1) // 4
    split_sizes = [dim] * 4
    r, i, j, k = torch.split(kernel, split_sizes, dim=1)
    r2 = torch.cat([r, -i, -j, -k], dim=1)  # 0, 1, 2, 3
    i2 = torch.cat([i, r, -k, j], dim=1)  # 1, 0, 3, 2
    j2 = torch.cat([j, k, r, -i], dim=1)  # 2, 3, 0, 1
    k2 = torch.cat([k, -j, i, r], dim=1)  # 3, 2, 1, 0
    hamilton = torch.cat([r2, i2, j2, k2], dim=0).to(device)
    
    assert kernel.size(1) == hamilton.size(1)
    return hamilton

'''Quaternion graph neural networks! QGNN layer for other downstream tasks!'''
class QGCNLayer_v2(Module):
    def __init__(self, in_features, out_features, act=torch.tanh, device = device, swap = True):
        super(QGCNLayer_v2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.act = act
        self.swap = swap
        self.weight = Parameter(torch.FloatTensor(self.in_features//4, self.out_features))
        self.reset_parameters()
        self.bn = torch.nn.BatchNorm1d(out_features)
        
        self.left_matrix = t_o(in_features).to(device)
        self.right_matrix = t_o(out_features).to(device)

    def reset_parameters(self):
        stdv = math.sqrt(6.0 / (self.weight.size(0) + self.weight.size(1)))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        hamilton = make_quaternion_mul(self.weight)
        if self.swap:
            hamilton = self.left_matrix.T @ hamilton @ self.right_matrix
        support = torch.mm(input, hamilton)  # Hamilton product, quaternion multiplication!
        output = torch.spmm(adj, support)
        output = self.bn(output)  # using act torch.tanh with BatchNorm can produce competitive results
        return self.act(output)

'''Quaternion graph neural networks! QGNN layer for node and graph classification tasks!'''
class QGCNLayer(Module):
    def __init__(self, in_features, out_features, dropout, quaternion_ff=True , swap = True, act=None, device = device):
        super(QGCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quaternion_ff = quaternion_ff 
        self.act = act
        self.dropout = nn.Dropout(dropout)
        self.bn = torch.nn.BatchNorm1d(out_features)
        self.swap = swap
        
        if self.swap:
            self.left_matrix = t_o(in_features).to(device)
            self.right_matrix = t_o(out_features).to(device)
        
        if self.quaternion_ff:
            self.weight = Parameter(torch.FloatTensor(self.in_features//4, self.out_features))
        else:
            self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = math.sqrt(6.0 / (self.weight.size(0) + self.weight.size(1)))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, double_type_used_in_graph=False):
        x = self.dropout(input) # Current Pytorch 1.5.0 doesn't support Dropout for sparse matrix
        if self.quaternion_ff:
            hamilton = make_quaternion_mul(self.weight)
            if self.swap:
                hamilton = self.left_matrix.T @ hamilton @ self.right_matrix
            if double_type_used_in_graph:  # to deal with scalar type between node and graph classification tasks
                hamilton = hamilton.double()

            support = torch.mm(x, hamilton)  # Hamilton product, quaternion multiplication!
        else:
            support = torch.mm(x, self.weight)

        if double_type_used_in_graph: #to deal with scalar type between node and graph classification tasks, caused by pre-defined feature inputs
            support = support.double()
            
        print(len(adj), len(support))
        output = torch.spmm(adj, support)

        # output = self.bn(output) # should tune whether using BatchNorm or Dropout

        if self.act is not None:
            output = self.act(output)
            
        return output
    
class QGCN(Module):
    def __init__(self, n_feat, n_class, n_layer, agg_hidden, fc_hidden, dropout):
        super(QGCN, self).__init__()
        
        self.n_layer = n_layer
        self.dropout = dropout
        
        # Graph convolution layer
        self.graph_convolution_layers = []
        for i in range(n_layer):
           if i == 0:
             self.graph_convolution_layers.append(QGCNLayer(n_feat, agg_hidden, dropout))
           else:
             self.graph_convolution_layers.append(QGCNLayer(agg_hidden, agg_hidden, dropout))
        
        # Fully-connected layer
        self.fc1 = nn.Linear(agg_hidden, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, n_class)
    
    def forward(self, x, adj):

        for i in range(self.n_layer):
           # Graph convolution layer
           x = F.relu(self.graph_convolution_layers[i](x, adj))
                      
           # Dropout
           if i != self.n_layer - 1:
             x = F.dropout(x, p=self.dropout, training=self.training)
        
        
        # Fully-connected layer
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x))

        return x
        
    def __repr__(self):
        layers = ''
        
        for i in range(self.n_layer):
            layers += str(self.graph_convolution_layers[i]) + '\n'
        layers += str(self.fc1) + '\n'
        layers += str(self.fc2) + '\n'
        return layers