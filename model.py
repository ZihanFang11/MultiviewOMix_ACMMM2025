import torch
import torch.nn as nn
import torch.nn.functional as F
import sys


class GCNLayer(nn.Module):
    """
    Graph Convolutional Network layer class.
    """
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, X, A):
        A_hat = A + torch.eye(A.size(0)).to(A.device)
        D_hat = torch.diag(torch.sum(A_hat, dim=1))
        D_hat_inv_sqrt = torch.inverse(torch.sqrt(D_hat))
        A_norm = torch.mm(torch.mm(D_hat_inv_sqrt, A_hat), D_hat_inv_sqrt)
        out = torch.mm(A_norm, X)
        out = self.linear(out)
        return out


class GCN(nn.Module):
    """
    Graph Convolutional Network model comprising two GCN layers.
    """
    def __init__(self, out_features, class_num):
        super(GCN, self).__init__()
        self.gcn1 = GCNLayer(out_features, out_features)
        self.gcn2 = GCNLayer(out_features, class_num)

    def forward(self, X, A):
        X = F.softplus(self.gcn2(self.gcn1(X, A), A))
        return X


class FusionLayer(nn.Module):
    """
    Fusion layer for combining embeddings from multiple views.
    """
    def __init__(self, num_views, fusion_type, in_size, hidden_size=32):
        super(FusionLayer, self).__init__()
        self.fusion_type = fusion_type
        if self.fusion_type == 'weight':
            self.weight = nn.Parameter(torch.ones(num_views) / num_views, requires_grad=True)
        if self.fusion_type == 'attention':
            self.encoder = nn.Sequential(
                nn.Linear(in_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 32, bias=False),
                nn.Tanh(),
                nn.Linear(32, 1, bias=False)
            )

    def forward(self, emb_list, weight):
        if self.fusion_type == "average":
            common_emb = sum(emb_list.values()) / len(emb_list)
        elif self.fusion_type == "weight":
            common_emb = sum([w * e for e, w in zip(weight, emb_list)])
        else:
            sys.exit("Please using a correct fusion type")
        return common_emb



class MLP(nn.Module):
    """
    Multi-layer Perceptron for processing node features.
    """
    def __init__(self, dims, num_classes):
        super(MLP, self).__init__()
        self.num_layers = len(dims)
        self.net = nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.net.append(nn.Linear(dims[i], dims[i + 1]))
            self.net.append(nn.ReLU())
            self.net.append(nn.Dropout(0.1))
        self.net.append(nn.Linear(dims[self.num_layers - 1], num_classes))
        self.net.append(nn.Softplus())

    def forward(self, x):
        h = self.net[0](x)
        for i in range(1, len(self.net)):
            h = self.net[i](h)
        return h


class MOCD(nn.Module):
    def __init__(self, num_views, dims, num_classes, gamma=0.5, fusion_type='average'):
        super(MOCD, self).__init__()
        self.num_views = num_views
        self.MSAN = nn.ModuleList([MLP(dims[i], num_classes) for i in range(num_views)])
        self.APN = nn.ModuleList([MLP(dims[i], num_classes) for i in range(num_views)])
        self.GCNs = nn.ModuleList([GCN(dims[i][0], num_classes) for i in range(num_views)])
        self.fusionlayer = FusionLayer(num_views, fusion_type, num_classes, hidden_size=64)
        self.gamma = nn.Parameter(torch.FloatTensor([gamma]), requires_grad=True)
        self.weight = torch.zeros(num_views, 1)
    def forward(self, X, A):
        view_specific_assignment = dict()
        for v in range(self.num_views):
            view_specific_assignment[v] = self.gamma * self.MSAN[v](X[v]) + (1 - self.gamma) * self.GCNs[v](X[v], A[v])
        return view_specific_assignment, self.fusionlayer(view_specific_assignment, self.weight)

    def agument(self, X):
        view_specific_assignment_Omix = dict()
        for v in range(self.num_views):
            view_specific_assignment_Omix[v] = self.APN[v](X[v])
        return view_specific_assignment_Omix

    def inference(self, X, A):
        view_specific_assignment = dict()
        for v in range(self.num_views):
            view_specific_assignment[v] =  self.gamma * self.MSAN[v](X[v]) + (1 - self.gamma) * self.GCNs[v](X[v], A[v])
        return self.fusionlayer(view_specific_assignment, self.weight)

