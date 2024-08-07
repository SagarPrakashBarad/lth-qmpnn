import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

"""@Dai Quoc Nguyen"""
'''Make a Hamilton matrix for quaternion linear transformations'''
def make_quaternion_mul(kernel):
    """" The constructed 'hamilton' W is a modified version of the quaternion representation,
        thus doing tf.matmul(Input,W) is equivalent to W * Inputs. """
    dim = kernel.size(1)//4
    r, i, j, k = torch.split(kernel, [dim, dim, dim, dim], dim=1)
    r2 = torch.cat([r, -i, -j, -k], dim=0)  # 0, 1, 2, 3
    i2 = torch.cat([i, r, -k, j], dim=0)  # 1, 0, 3, 2
    j2 = torch.cat([j, k, r, -i], dim=0)  # 2, 3, 0, 1
    k2 = torch.cat([k, -j, i, r], dim=0)  # 3, 2, 1, 0
    hamilton = torch.cat([r2, i2, j2, k2], dim=1)
    assert kernel.size(1) == hamilton.size(1)
    return hamilton

'''Quaternion graph neural networks! QGNN layer for other downstream tasks!'''
class QGCNLayer_v2(Module):
    def __init__(self, in_features, out_features, act=torch.tanh):
        super(QGCNLayer_v2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.act = act
        self.weight = Parameter(torch.FloatTensor(self.in_features//4, self.out_features))
        self.reset_parameters()
        self.bn = torch.nn.BatchNorm1d(out_features)

    def reset_parameters(self):
        stdv = math.sqrt(6.0 / (self.weight.size(0) + self.weight.size(1)))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        hamilton = make_quaternion_mul(self.weight)
        
        support = torch.mm(input, hamilton)  # Hamilton product, quaternion multiplication!
        output = torch.spmm(adj, support)
        output = self.bn(output)  # using act torch.tanh with BatchNorm can produce competitive results
        return self.act(output)

'''Quaternion graph neural networks! QGNN layer for node and graph classification tasks!'''
class QGCNLayer(Module):
    def __init__(self, in_features, out_features, dropout, quaternion_ff=True, act=None):
        super(QGCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quaternion_ff = quaternion_ff 
        self.act = act
        self.dropout = nn.Dropout(dropout)
        self.bn = torch.nn.BatchNorm1d(out_features)
        #
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
            if double_type_used_in_graph:  # to deal with scalar type between node and graph classification tasks
                hamilton = hamilton.double()

            support = torch.mm(x, hamilton)  # Hamilton product, quaternion multiplication!
        else:
            support = torch.mm(x, self.weight)

        if double_type_used_in_graph: #to deal with scalar type between node and graph classification tasks, caused by pre-defined feature inputs
            support = support.double()

        output = torch.spmm(adj, support)

        # output = self.bn(output) # should tune whether using BatchNorm or Dropout

        if self.act is not None:
            output = self.act(output)
            
        return output
    
    
"""Graph Attention Network using Quaternion multiplication"""
class QGATLayer_v2(nn.Module):
    def __init__(self, in_features, out_features, alpha, concat=True):
        super(QGATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

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
class QGATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, quaternion_ff = True, alpha = 0.2, concat=True,):
        super(QGATLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.quaternion_ff = quaternion_ff

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
        
        
class QGATMultiHeadLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, num_heads=4, quaternion_ff=True, concat=True):
        super(QGATMultiHeadLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.num_heads = num_heads
        self.concat = concat
        self.quaternion_ff = quaternion_ff

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