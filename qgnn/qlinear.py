
import torch
import torch.nn as nn
import numpy as np

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
def make_quaternion_mul(kernel, quat_swap =True):
    dim = kernel.size(1) // 4
    split_sizes = [dim] * 4
    r, i, j, k = torch.split(kernel, split_sizes, dim=1)
    r2 = torch.cat([r, -i, -j, -k], dim=0)  # 0, 1, 2, 3
    i2 = torch.cat([i, r, -k, j], dim=0)  # 1, 0, 3, 2
    j2 = torch.cat([j, k, r, -i], dim=0)  # 2, 3, 0, 1
    k2 = torch.cat([k, -j, i, r], dim=0)  # 3, 2, 1, 0
    hamilton = torch.cat([r2, i2, j2, k2], dim=1)
    assert kernel.size(1) == hamilton.size(1)
    return hamilton
"""

class QLinear(nn.Module):
    def __init__(self, in_features, out_features, swap= True, bias=True, device = device):
        super(QLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.swap = swap
        self.device = device
        self.weight = nn.Parameter(torch.Tensor(in_features//4, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.left_matrix = t_o(in_features).to(device)
        self.right_matrix = t_o(out_features).to(device)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        hamilton = make_quaternion_mul(self.weight)
        if self.swap:
            hamilton = torch.matmul(torch.matmul(self.left_matrix.T, hamilton), self.right_matrix).to(device)
        support = torch.mm(input, hamilton)
        if self.bias is not None:
            support += self.bias.view(1, -1)
        return support
    
