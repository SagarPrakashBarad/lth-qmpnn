import torch

from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import GCNConv, SAGEConv, GATConv

from datasets import preprocess_batch
from qgnn import QGCNLayer, QGATLayer, QSAGELayer

class QGNN(torch.nn.Module):
    def __init__(self, 
                 input_dim,
                 num_classes,
                 model:         dict(help='Model to use', choices=['gcn', 'sage', 'gat']) = 'gcn',
                 hidden_dim:    dict(help='Hidden dimension') = 16,
                 dropout:       dict(help='Dropout rate') = 0.5,
                 num_layers:    dict(help='Number of layers') = 2,
                 quat:          dict(help='Use quaternion representation') = False,
                 swap:          dict(help='unitary swap') = True    
            ):
        super(QGNN, self).__init__()
        
        self.model = model
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_layers = num_layers
        self.quat = quat
        self.swap = swap
        
        self.convs = torch.nn.ModuleList()
        
        
        if self.model == 'gcn':
            self.convs.append(QGCNLayer(input_dim, hidden_dim, dropout, quat, swap))
            for _ in range(num_layers - 2):
                self.convs.append(QGCNLayer(hidden_dim, hidden_dim, dropout, quat, swap))
            self.convs.append(QGCNLayer(hidden_dim, num_classes, dropout, quaternion_ff=False))
        
        elif self.model == 'gat':
            self.convs.append(QGATLayer(input_dim, hidden_dim, dropout, quat, swap))
            for _ in range(num_layers - 2):
                self.convs.append(QGATLayer(hidden_dim, hidden_dim, dropout, quat, swap))
            self.convs.append(QGATLayer(hidden_dim, num_classes, dropout, quaternion_ff=False))
            
        elif self.model == 'sage':
            self.convs.append(QSAGELayer(input_dim, hidden_dim, dropout, quat))
            for _ in range(num_layers - 2):
                self.convs.append(QSAGELayer(hidden_dim, hidden_dim, dropout, quat))
            self.convs.append(QSAGELayer(hidden_dim, num_classes, dropout, quaternion_ff=False))
    
        """        
        if self.model == 'gcn':
            self.convs.append(GCNConv(input_dim, hidden_dim, dropout))
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_dim, hidden_dim, dropout))
            self.convs.append(GCNConv(hidden_dim, num_classes, dropout))
            
        elif self.model == 'gat':
            self.convs.append(GATConv(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(hidden_dim, hidden_dim))
            self.convs.append(GATConv(hidden_dim, num_classes))
            
        elif self.model == 'sage':
            self.convs.append(SAGEConv(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.convs.append(SAGEConv(hidden_dim, num_classes))
        
        else:
            raise ValueError('Model not supported')
        
        
    
    """
    def forward(self, x, adj_t):
        for conv in self.convs:
            x = conv(x, adj_t)
        return x

            
class NodeClassifier(torch.nn.Module):
    def __init__(self, 
                 input_dim,
                 num_classes,
                 model:         dict(help='Model to use', choices=['gcn', 'sage', 'gat']) = 'gcn',
                 hidden_dim:    dict(help='Hidden dimension') = 64,
                 dropout:       dict(help='Dropout rate') = 0.5,
                 num_layers:    dict(help='Number of layers') = 2,
                 quat:          dict(help='Use quaternion representation') = False,
                 swap:          dict(help='Unitary swap') = True    
            ):
        super(NodeClassifier, self).__init__()
        if num_classes % 4 != 0:
            num_classes += 4 - num_classes % 4
        
        self.gnn = QGNN(input_dim, num_classes, model, hidden_dim, dropout, num_layers, quat, swap)
        self.softmax = F.log_softmax
        
    def forward(self, data):
        x, adj_t = data.x, data.adj_t
        x = self.gnn(x, adj_t)
        return self.softmax(x, dim=1)
    
    def training_step(self, batch):
        # x, adj_t, y = batch.x, batch.adj_t, batch.y
        out = self.forward(batch)
        loss = self.loss(out, batch.y)
        acc = self.accuracy(out, batch.y)
        return loss, acc
    
    def validation_step(self, batch):
        return self.training_step(batch)
    
    @staticmethod
    def accuracy(out, y):
        return (out.argmax(dim=1) == y).sum().item() / y.size(0)
    
    @staticmethod
    def loss(out, y):
        return F.nll_loss(out, y)
        

class GraphClassifier(torch.nn.Module):
    def __init__(self, 
                 input_dim,
                 num_classes,
                 model:         dict(help='Model to use', choices=['gcn', 'sage', 'gat']) = 'gcn',
                 hidden_dim:    dict(help='Hidden dimension') = 4,
                 dropout:       dict(help='Dropout rate') = 0.5,
                 num_layers:    dict(help='Number of layers') = 3,
                 quat:          dict(help='Use quaternion representation') = False,
                 swap:          dict(help='Unitary swap') = True      
            ):
        super(GraphClassifier, self).__init__()
        
        # note input dim is multiplied by 4 to account for quaternion representation
        if num_classes % 4 != 0:
            num_classes += 4 - num_classes % 4
        #if quat:
        #    input_dim *= 4
        self.gnn = QGNN(input_dim, num_classes, model, hidden_dim, dropout, num_layers, quat, swap)
        self.softmax = F.log_softmax
        self.quat = quat
        
        
    def forward(self, data):
        x, adj_t = data.x, data.adj_t
        x = self.gnn(x, adj_t)
        x = global_mean_pool(x, data.batch)
        
        x = self.softmax(x, dim=1)
        return x
    
    def training_step(self, data, optimizer):
        train_loader = DataLoader(data[data.train_mask], batch_size=64, shuffle=True)
        correct, losses = 0, 0
        for graph in train_loader:
            graph = preprocess_batch(graph)
            optimizer.zero_grad()
            out = self.forward(graph)
            correct += self.accuracy(out, graph.y)
            loss = self.loss(out, graph.y)
            losses += loss.item()
            loss.backward()
            optimizer.step()
        
        acc = correct / len(train_loader)
        losses /= len(train_loader)

        metrics = {'train/loss': losses, 'train/acc': acc}
        return metrics
    
    @torch.no_grad()
    def validation_step(self, data):
        val_loader = DataLoader(data[data.val_mask], batch_size=64, shuffle=False)
        correct, losses = 0, 0
        for graph in val_loader:
            graph = preprocess_batch(graph)
            out = self.forward(graph)
            correct += self.accuracy(out, graph.y)
            loss = self.loss(out, graph.y)
            losses += loss.item()
        
        acc = correct / len(val_loader)
        losses /= len(val_loader)

        metrics = {'val/loss': losses, 'val/acc': acc}
        return metrics
    
    @staticmethod
    def accuracy(out, y):
        return (out.argmax(dim=1) == y).sum().item() / y.size(0)
    
    @staticmethod
    def loss(out, y):
        return torch.nn.CrossEntropyLoss()(out, y)

class LinkPredictor(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 out_dim = 64,
                 model:         dict(help='Model to use', choices=['gcn', 'sage', 'gat']) = 'gcn',
                 hidden_dim:    dict(help='Hidden dimension') = 64,
                 dropout:       dict(help='Dropout rate') = 0.5,
                 num_layers:    dict(help='Number of layers') = 2,
                 quat:          dict(help='Use quaternion representation') = False,
                 swap:          dict(help='Unitary swap') = True  
    ):
        super(LinkPredictor, self).__init__()
        
        #if quat:
        #    input_dim *= 4
        self.gnn = QGNN(input_dim, out_dim, model, hidden_dim, dropout, num_layers, quat, swap)
        self.quat = quat
        
    def encode(self, x, adj_t):
        return self.gnn(x, adj_t)
        
    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)
    
    def training_step(self, data, optimizer):
        optimizer.zero_grad()
        data = preprocess_batch(data)
  
        z = self.encode(data.x, data.adj_t)

        # negative sampling
        neg_edge_index = negative_sampling(
            edge_index=data.edge_index, 
            num_nodes=data.num_nodes, 
            num_neg_samples=data.edge_label_index.size(1), 
            method='sparse')
        
        edge_label_index = torch.cat(
            [data.edge_label_index, neg_edge_index], 
            dim=-1,
        )
        edge_label = torch.cat([
            data.edge_label,
            data.edge_label.new_zeros(neg_edge_index.size(1))
        ], dim=0)
        
        out = self.decode(z, edge_label_index).view(-1)
        loss = self.loss(out, edge_label)
        loss.backward()
        optimizer.step()
        roc_auc = self.roc_auc(out, edge_label)
        return loss, roc_auc
    
    @torch.no_grad()
    def validation_step(self, data):
        data = preprocess_batch(data)
        z = self.encode(data.x, data.adj_t)
        out = self.decode(z, data.edge_label_index).view(-1)
        loss = self.loss(out, data.edge_label)
        roc_auc = self.roc_auc(out, data.edge_label)
        return loss, roc_auc
    
    @torch.no_grad()
    def test_step(self, data):
        return self.validation_step(data)
        
    @staticmethod
    def roc_auc(out, label):
        return roc_auc_score(label.cpu().detach().numpy(), out.cpu().detach().numpy())
    
    @staticmethod
    def loss(out, label):
        return torch.nn.BCEWithLogitsLoss()(out, label)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = NodeClassifier(1433, 7, model='gcn', hidden_dim=16, dropout=0.5, num_layers=2, quat=True)
    print(count_parameters(model))
    

        