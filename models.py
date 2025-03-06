import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import Linear, HeteroConv, SAGEConv, GCNConv, GATConv
from torch_geometric.data import DataLoader

from torch_geometric.nn.conv.gcn_conv import gcn_norm
import numpy as np
from torch_geometric.nn import MessagePassing, APPNP

from deeprobust.graph.defense import GCNJaccard
import scipy.sparse as sp

from tqdm import tqdm


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout_rate = dropout_rate

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout_rate)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads, dropout_rate):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=0.6)
        # On the Pubmed dataset, use `heads` output heads in `conv2`.
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1,
                             concat=False, dropout=0.6)
        self.dropout_rate = dropout_rate

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x)


class NodeFeatureModel(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate):
        super(NodeFeatureModel, self).__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.dropout_rate = dropout_rate

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    def forward(self, data):
        x = data.x
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class _NodeFeatureModel(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate):
        super(_NodeFeatureModel, self).__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.dropout_rate = dropout_rate

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    def forward(self, flow_features):
        x = self.fc1(flow_features)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CombinedModel(nn.Module):
    def __init__(self, gcn_model, node_feature_model, out_channels):
        super(CombinedModel, self).__init__()
        self.gcn_model = gcn_model
        self.node_feature_model = node_feature_model

        self.beta = nn.Parameter(torch.tensor(0.5), requires_grad=True)

    def reset_parameters(self):
        self.node_feature_model.reset_parameters()
        self.gcn_model.reset_parameters()
        self.beta = nn.Parameter(torch.tensor(0.5), requires_grad=True)

    def forward(self, data):
        # x, edge_index = data.x, data.edge_index
        gcn_output = self.gcn_model(data)
        node_feature_output = self.node_feature_model(data)

        combined_output = self.beta * gcn_output + (1 - self.beta) * node_feature_output

        return combined_output


class CombinedModelVector(nn.Module):
    def __init__(self, gcn_model, node_feature_model, out_channels):
        super(CombinedModelVector, self).__init__()
        self.out_channels = out_channels
        self.gcn_model = gcn_model
        self.node_feature_model = node_feature_model
        # self.beta = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.full((out_channels,), 0.5))

    def reset_parameters(self):
        self.node_feature_model.reset_parameters()
        self.gcn_model.reset_parameters()
        # self.beta = nn.Parameter(torch.full((self.out_channels,), 0.5))

    def forward(self, data):
        # x, edge_index = data.x, data.edge_index
        gcn_output = self.gcn_model(data)
        node_feature_output = self.node_feature_model(data)

        combined_output = self.beta * gcn_output + (1 - self.beta) * node_feature_output
        return combined_output


class InterLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate):
        super(InterLayer, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout_rate = dropout_rate

        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)

        # Define learnable parameters beta1 and beta2
        self.beta2 = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.beta1 = nn.Parameter(torch.tensor(0.5), requires_grad=True)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x1 = x.clone()

        x = self.conv1(x, edge_index)
        x1 = self.fc1(x1)

        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout_rate)

        x1 = F.relu(x1)
        x1 = F.dropout(x1, training=self.training, p=self.dropout_rate)

        # stacked_l1 = torch.stack((x, x1), dim=0)
        # x = torch.mean(stacked_l1, dim=0)
        stacked_l1 = torch.stack((self.beta1 * x, (1 - self.beta1) * x1), dim=0)
        x = torch.sum(stacked_l1, dim=0)

        x = self.conv2(x, edge_index)
        x1 = self.fc2(x1)

        # x = F.relu(x)
        # x1 = F.relu(x1)

        # stacked_l2 = torch.stack((x, x1), dim=0)
        # x = torch.mean(stacked_l2, dim=0)
        stacked_l2 = torch.stack((self.beta2 * x, (1 - self.beta2) * x1), dim=0)
        x = torch.sum(stacked_l2, dim=0)

        return F.log_softmax(x, dim=1)


class GPR_Prop(MessagePassing):
    """
    Copied from the source: https://github.com/jianhao2016/GPRGNN/blob/master/src/GNN_models.py#L225


    propagation class for GPR_GNN
    """

    def __init__(self, K, alpha, Init, Gamma=None, bias=True, **kwargs):
        super(GPR_Prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha
        self.Gamma = Gamma

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            # SGC-like, note that in this case, alpha has to be a integer. It means where the peak at when
            # initializing GPR weights.
            TEMP = 0.0 * np.ones(K + 1)
            TEMP[alpha] = 1.0
        elif Init == 'PPR':
            # PPR-like
            TEMP = alpha * (1 - alpha) ** np.arange(K + 1)
            TEMP[-1] = (1 - alpha) ** K
        elif Init == 'NPPR':
            # Negative PPR
            TEMP = (alpha) ** np.arange(K + 1)
            TEMP = TEMP / np.sum(np.abs(TEMP))
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3 / (K + 1))
            TEMP = np.random.uniform(-bound, bound, K + 1)
            TEMP = TEMP / np.sum(np.abs(TEMP))
        elif Init == 'WS':
            # Specify Gamma
            TEMP = Gamma

        self.temp = nn.Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        if self.Init == 'SGC':
            self.temp.data[self.alpha] = 1.0
        elif self.Init == 'PPR':
            for k in range(self.K + 1):
                self.temp.data[k] = self.alpha * (1 - self.alpha) ** k
            self.temp.data[-1] = (1 - self.alpha) ** self.K
        elif self.Init == 'NPPR':
            for k in range(self.K + 1):
                self.temp.data[k] = self.alpha ** k
            self.temp.data = self.temp.data / torch.sum(torch.abs(self.temp.data))
        elif self.Init == 'Random':
            bound = np.sqrt(3 / (self.K + 1))
            torch.nn.init.uniform_(self.temp, -bound, bound)
            self.temp.data = self.temp.data / torch.sum(torch.abs(self.temp.data))
        elif self.Init == 'WS':
            self.temp.data = self.Gamma

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)

        hidden = x * (self.temp[0])
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[k + 1]
            hidden = hidden + gamma * x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)


class GPRGNN(torch.nn.Module):
    """
    Copied from the source: https://github.com/jianhao2016/GPRGNN/blob/master/src/GNN_models.py#L225

    """

    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate, args):
        super(GPRGNN, self).__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)

        if args.ppnp == 'PPNP':
            self.prop1 = APPNP(args.K, args.alpha)
        elif args.ppnp == 'GPR_prop':
            self.prop1 = GPR_Prop(args.K, args.alpha, args.Init, args.Gamma)

        self.Init = args.Init
        self.dprate = dropout_rate
        self.dropout = 0.5

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)


class ModifiedGCNJaccard(GCNJaccard):
    def drop_dissimilar_edges(self, features, adj, threshold):
        """Modified version of drop_dissimilar_edges with proper handling of zero-degree nodes."""
        if not sp.issparse(adj):
            adj = sp.csr_matrix(adj)

        adj_triu = sp.triu(adj, format='csr')
        rows, cols = adj_triu.nonzero()

        # Calculate similarity scores
        scores = []
        for i, j in zip(rows, cols):
            # Get neighbors
            neighbors_i = adj[i].nonzero()[1]
            neighbors_j = adj[j].nonzero()[1]

            # Handle zero-degree nodes
            if len(neighbors_i) == 0 or len(neighbors_j) == 0:
                # Option 1: Assign minimum similarity
                scores.append(0.0)
                continue

                # Option 2: Use feature similarity instead
                # feature_sim = np.dot(features[i], features[j]) / (np.linalg.norm(features[i]) * np.linalg.norm(features[j]))
                # scores.append(feature_sim)
                # continue

            # Calculate Jaccard similarity
            intersection = len(np.intersect1d(neighbors_i, neighbors_j))
            union = len(np.union1d(neighbors_i, neighbors_j))

            scores.append(intersection / union)

        # Drop edges with similarity scores below threshold
        adj_triu = adj_triu.tolil()
        for idx, (i, j) in enumerate(zip(rows, cols)):
            if scores[idx] < threshold:
                adj_triu[i, j] = 0

        # Make symmetric
        adj_triu = adj_triu.tocsr()
        adj_final = adj_triu + adj_triu.T

        return adj_final

    def fit(self, features, adj, labels, idx_train, idx_val=None, train_iters=200, threshold=0.01, **kwargs):
        """Modified fit method using the new drop_dissimilar_edges implementation."""
        self.threshold = threshold
        modified_adj = self.drop_dissimilar_edges(features, adj, threshold)
        return super(GCNJaccard, self).fit(features, modified_adj, labels, idx_train, idx_val, train_iters=train_iters,
                                           **kwargs)


class HeteroGNN(torch.nn.Module):
    """
    Copied from the source: https://github.com/PacktPublishing/Hands-On-Graph-Neural-Networks-Using-Python/blob/main/Chapter16/chapter16.ipynb

    """

    def __init__(self, dim_h, dim_out, num_layers):
        super(HeteroGNN, self).__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('host', 'to', 'flow'): SAGEConv((-1, -1), dim_h),
                ('flow', 'to', 'host'): SAGEConv((-1, -1), dim_h),
            }, aggr='sum')
            self.convs.append(conv)

        self.lin = Linear(dim_h, dim_out)

    def forward(self, batch):
        x_dict = batch.x_dict
        edge_index_dict = batch.edge_index_dict
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
        return self.lin(x_dict['flow'])


class CombinedHeteroModel(nn.Module):
    def __init__(self, hetero_gnn, node_feature_model, out_channels):
        super(CombinedHeteroModel, self).__init__()
        self.hetero_gnn = hetero_gnn
        self.node_feature_model = node_feature_model
        self.beta = nn.Parameter(torch.tensor(0.5), requires_grad=True)  # Learnable weight

    def reset_parameters(self):
        self.hetero_gnn.reset_parameters()
        self.node_feature_model.reset_parameters()
        self.beta = nn.Parameter(torch.tensor(0.5), requires_grad=True)

    def forward(self, batch):
        x_dict = batch.x_dict
        edge_index_dict = batch.edge_index_dict
        flow_features = batch['flow'].x  # Extract features for 'flow' node type

        # HeteroGNN processing (returns dictionary of embeddings)
        hetero_output = self.hetero_gnn(x_dict, edge_index_dict)  # Get flow node embeddings
        # Ensure data.x corresponds to the "flow" node type for NodeFeatureModel
        node_feature_output = self.node_feature_model(flow_features)  # Process homogeneous features

        # Weighted combination of both outputs
        combined_output = self.beta * hetero_output + (1 - self.beta) * node_feature_output

        return combined_output
