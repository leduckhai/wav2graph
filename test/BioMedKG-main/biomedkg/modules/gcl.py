import torch
import torch.nn.functional as F
from torch_geometric.nn.inits import uniform
from torch_geometric.utils import dropout_edge, mask_feature

# reference from PyGCL
class DGI(torch.nn.Module):
    def __init__(self, encoder, hidden_dim):
        super().__init__()
        self.encoder = encoder
        self.project = torch.nn.Linear(hidden_dim, hidden_dim)
        uniform(hidden_dim, self.project.weight)

    @staticmethod
    def corruption(x, edge_index):
        return x[torch.randperm(x.size(0))], edge_index
    
    @staticmethod
    def summary(z:torch.tensor):
        return torch.sigmoid(z.mean(dim=0, keepdim=True))

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        g = self.project(self.summary(z))
        zn = self.encoder(*self.corruption(x, edge_index))
        return z, g, zn
    
    
# reference from PyGCL
class GRACE(torch.nn.Module):
    def __init__(self, encoder, hidden_dim, proj_dim):
        super().__init__()
        self.encoder = encoder

        self.fc1 = torch.nn.Linear(hidden_dim, proj_dim)
        self.fc2 = torch.nn.Linear(proj_dim, hidden_dim)

    def forward(self, x, edge_index):
        x1, _ = mask_feature(x, p=0.4, mode="all")
        x2, _ = mask_feature(x, p=0.4, mode="all")
        edge_index1, _ = dropout_edge(edge_index, p=0.4)
        edge_index2, _ = dropout_edge(edge_index, p=0.4)
        z = self.encoder(x, edge_index)
        z1 = self.encoder(x1, edge_index1)
        z2 = self.encoder(x2, edge_index2)
        return z, z1, z2

    def project(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)


class GGD(torch.nn.Module):
    """
    Implementation of https://arxiv.org/abs/2206.01535
    """
    def __init__(self, encoder, hidden_dim, n_proj, aug_p):
        super().__init__()
        self.encoder = encoder
        self.p = aug_p

        mlp = list()
        for _ in range(n_proj):
            mlp.append(torch.nn.Linear(hidden_dim, hidden_dim))
        self.mlp = torch.nn.ModuleList(mlp)

    @staticmethod
    def corruption(x, edge_index):
        return x[torch.randperm(x.size(0))], edge_index
    
    def forward(self, x, edge_index):
        if torch.rand(1).item() < self.p:
            x, _ = mask_feature(x, p=0.4, mode="all")
            edge_index, _ = dropout_edge(edge_index, p=0.4)
        
        pos_z = self.encoder(x, edge_index)
        neg_z = self.encoder(*self.corruption(x, edge_index))

        pos_h, neg_h = pos_z, neg_z

        for layer in self.mlp[:-1]:
            pos_h = F.relu(layer(pos_h))
            neg_h = F.relu(layer(neg_h))
        
        pos_h = self.mlp[-1](pos_h)
        neg_h = self.mlp[-1](neg_h)

        pos_h = pos_h.sum(1)
        neg_h = neg_h.sum(1)

        return pos_h, neg_h