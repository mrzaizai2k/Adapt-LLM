import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GINConv, AttentionalAggregation

class GNNGraphEncoder(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim=128,
        embedding_dim=500,
        num_layers=3,
        dropout=0.1,
    ):
        super().__init__()

        self.convs = nn.ModuleList()

        for i in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp))

        self.pool = AttentionalAggregation(
            gate_nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        )

        self.project = nn.Linear(hidden_dim, embedding_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        graph_emb = self.pool(x, batch)
        return self.project(graph_emb)
