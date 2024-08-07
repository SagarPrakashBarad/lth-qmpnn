import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

"""
Base paper: https://arxiv.org/abs/1710.10903
"""
    
"""Graph Attention Network using Quaternion multiplication"""
class QGATLayer_v2(Module):
    def __init__(self, in_features, out_features, alpha, concat=True, swap=False, device = device):
        super(QGATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.swap = swap
        
        self.left_matrix = t_o(in_features).to(device)
        self.right_matrix = t_o(out_features).to(device)

        self.W = Parameter(torch.zeros(size=(in_features//4, out_features)))
        self.a = Parameter(torch.zeros(size=(2 * out_features, 1)))
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.bn = nn.BatchNorm1d(out_features)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.data)
        nn.init.xavier_uniform_(self.a.data)
        
    def forward(self, input, adj):
        
        hamilton = make_quaternion_mul(self.W)
        if self.swap:
            hamilton = self.left_matrix.T @ hamilton @ self.right_matrix
        h = torch.mm(input, hamilton)

        N = h.size()[0]
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        adj_dense = adj.to_dense()
        mask = (adj_dense > 0)
        attention = torch.where(mask, e, zero_vec)

        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        h_prime = torch.matmul(attention, h)

        h_prime = self.bn(h_prime)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

"""Graph Attention Network using Quaternion multiplication"""
class QGATLayer(Module):
    def __init__(self, in_features, out_features, dropout, quaternion_ff = True , swap = False, alpha = 0.2, concat=True, device = device):
        super(QGATLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.quaternion_ff = quaternion_ff
        
        self.swap = swap
        
        self.left_matrix = t_o(in_features).to(device)
        self.right_matrix = t_o(out_features).to(device)

        if self.quaternion_ff:
            self.W = Parameter(torch.zeros(size=(in_features//4, out_features)))
        else:
            self.W = Parameter(torch.zeros(size=(in_features, out_features)))
            
        self.a = Parameter(torch.zeros(size=(2 * out_features, 1)))
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.bn = nn.BatchNorm1d(out_features)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.data)
        nn.init.xavier_uniform_(self.a.data)
        
    def forward(self, input, adj):
        
        if self.quaternion_ff:
            hamilton = make_quaternion_mul(self.W)
            if self.swap:
                hamilton = self.left_matrix.T @ hamilton @ self.right_matrix
            h = torch.mm(input, hamilton)
        else:
            h = torch.mm(input, self.W)

        N = h.size()[0]
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        adj_dense = adj.to_dense()
        mask = (adj_dense > 0)
        attention = torch.where(mask, e, zero_vec)

        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        h_prime = torch.matmul(attention, h)

        h_prime = self.bn(h_prime)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
        
        
class QGATMultiHeadLayer(Module):
    def __init__(self, in_features, out_features, dropout, alpha, num_heads=4, quaternion_ff=True, concat=True, swap=False, device = device):
        super(QGATMultiHeadLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.num_heads = num_heads
        self.concat = concat
        self.quaternion_ff = quaternion_ff
        
        self.swap = swap
        
        self.left_matrix = t_o(in_features).to(device)
        self.right_matrix = t_o(out_features).to(device)

        if self.quaternion_ff:
            self.W_heads = nn.ParameterList([nn.Parameter(torch.zeros(size=(self.in_features//4, self.out_features))) for _ in range(self.num_heads)])
        else:
            self.W_heads = nn.ParameterList([nn.Parameter(torch.zeros(size=(self.in_features, self.out_features))) for _ in range(self.num_heads)])

        self.a_heads = nn.ParameterList([nn.Parameter(torch.zeros(size=(2 * self.out_features, 1))) for _ in range(self.num_heads)])
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.bn = nn.BatchNorm1d(out_features)
    
    def reset_parameters(self):
        if self.quaternion_ff:
            stdv = math.sqrt(6.0 / (self.W_heads[0].size(0) + self.W_heads[0].size(1)))
            for W_head in self.W_heads:
                W_head.data.uniform_(-stdv, stdv)

            for a_head in self.a_heads:
                nn.init.xavier_uniform_(a_head.data, gain=1.414)

    def forward(self, input, adj):
        
        head_outputs = []

        for i in range(self.num_heads):
            
            if self.quaternion_ff:
                hamilton = make_quaternion_mul(self.W_heads[i])
                if self.swap:
                    hamilton = self.left_matrix.T @ hamilton @ self.right_matrix
                
                h = torch.mm(input, hamilton)
            else:
                h = torch.mm(input, self.W_heads[i])

            N = h.size()[0]
            a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
            e = nn.LeakyReLU(self.alpha)(torch.matmul(a_input, self.a_heads[i]).squeeze(2))

            zero_vec = -9e15 * torch.ones_like(e)
            adj_dense = adj.to_dense()
            mask = (adj_dense > 0)
            attention = torch.where(mask, e, zero_vec)

            attention = F.softmax(attention, dim=1)
            attention = F.dropout(attention, self.dropout, training=self.training)

            h_prime = torch.matmul(attention, h)
            head_outputs.append(h_prime)

        output = torch.cat(head_outputs, dim=1)

        if self.concat:
            return F.elu(output)
        else:
            return output
        

class QGAT(nn.Module):
    def __init__(self, n_feat, n_class, n_layer, agg_hidden, fc_hidden, dropout):
        super(QGAT, self).__init__()

        self.n_layer = n_layer
        self.dropout = dropout
        
        # Graph attention layer
        self.graph_attention_layers = []
        for i in range(self.n_layer):
          self.graph_attention_layers.append(QGATLayer(n_feat, agg_hidden, dropout))
                    
        # Fully-connected layer
        self.fc1 = nn.Linear(agg_hidden*n_layer, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, n_class)
        
    def forward(self, x, adj):
        
        # Dropout        
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Graph attention layer
        x = torch.cat([F.relu(att(x, adj)) for att in self.graph_attention_layers], dim=2)
        
        # Fully-connected layer
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x))
        
        return x

    def __repr__(self):
        layers = ''
        
        for i in range(self.n_layer):
            layers += str(self.graph_attention_layers[i]) + '\n'
        layers += str(self.fc1) + '\n'
        layers += str(self.fc2) + '\n'
        return layers