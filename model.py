import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv,SAGEConv
import torch.nn as nn
from arg_parser import parse_args
import torch_geometric.nn as geom_nn

#GraphNet 是一个继承自 torch.nn.Module 的类，表示一个图神经网络模型。

#它包含了初始化网络架构和定义前向传播过程的内容.
class GraphNet(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=128, mlp_hidden_channels=256, num_classes=1,heads=8):
        super(GraphNet, self).__init__()
        args = parse_args()
        self.droup_out = args.droup_out

        # self.conv1 = SAGEConv(num_node_features, hidden_channels)
        # self.conv2 = SAGEConv(hidden_channels, hidden_channels)

        self.conv1 = GATConv(num_node_features, hidden_channels // heads, heads=heads)
        self.conv2 = GATConv(hidden_channels, hidden_channels // heads, heads=heads)

        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels, mlp_hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=self.droup_out),
            nn.Linear(mlp_hidden_channels, num_classes)
        )

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.droup_out, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.droup_out, training=self.training)

        edge_features = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1)
        edge_prediction = self.mlp(edge_features)

        return edge_prediction.view(-1)


