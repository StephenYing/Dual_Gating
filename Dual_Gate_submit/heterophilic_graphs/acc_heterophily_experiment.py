import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch_scatter import scatter
from torch_geometric.nn import (
    GCNConv, GATConv, SAGEConv
)
from torch_geometric.datasets import WebKB, WikipediaNetwork, Actor
from torch_geometric.utils import to_undirected

import argparse

def get_data(name, split=0):
    path = f'../data/{name}'
    if name in ['chameleon', 'squirrel']:
        dataset = WikipediaNetwork(root=path, name=name)
    elif name in ['cornell', 'texas', 'wisconsin']:
        dataset = WebKB(path, name=name)
    elif name == 'film':
        dataset = Actor(root=path)
    else:
        raise ValueError(f'Unknown dataset: {name}')

    data = dataset[0]

    if name in ['chameleon', 'squirrel']:
        splits_file = np.load(f'{path}/{name}/geom_gcn/raw/{name}_split_0.6_0.2_{split}.npz')
    elif name in ['cornell', 'texas', 'wisconsin']:
        splits_file = np.load(f'{path}/{name}/raw/{name}_split_0.6_0.2_{split}.npz')
    elif name == 'film':
        splits_file = np.load(f'{path}/raw/{name}_split_0.6_0.2_{split}.npz')
    else:
        raise ValueError(f'No split file for {name}')

    train_mask = splits_file['train_mask']
    val_mask   = splits_file['val_mask']
    test_mask  = splits_file['test_mask']

    data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
    data.val_mask   = torch.tensor(val_mask,   dtype=torch.bool)
    data.test_mask  = torch.tensor(test_mask,  dtype=torch.bool)

    data.edge_index = to_undirected(data.edge_index)
    return data


class G2(nn.Module):
    def __init__(self, conv, p=2., conv_type='SAGE', activation=nn.ReLU()):
        super().__init__()
        self.conv = conv
        self.p = p
        self.activation = activation
        self.conv_type = conv_type

    def forward(self, X, edge_index):
        n_nodes = X.size(0)
        # aggregator pass
        if self.conv_type == 'GAT':
            X = F.elu(self.conv(X, edge_index)).view(n_nodes, -1, 4).mean(dim=-1)
        else:
            X = self.activation(self.conv(X, edge_index))

        # gating tau => local diffs
        row, col = edge_index
        diffs = (torch.abs(X[row] - X[col]) ** self.p).squeeze(-1)
        gg = scatter(diffs, row, dim=0, dim_size=X.size(0), reduce='mean')
        gg = torch.tanh(gg)
        return gg


class G2_GNN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayers,
                 conv_type='SAGE', p=2., drop_in=0, drop=0, use_gg_conv=True):
        super().__init__()
        self.conv_type = conv_type
        self.enc = nn.Linear(nfeat, nhid)
        self.dec = nn.Linear(nhid, nclass)
        self.drop_in = drop_in
        self.drop = drop
        self.nlayers = nlayers

        if conv_type == 'GCN':
            self.conv = GCNConv(nhid, nhid)
            if use_gg_conv:
                self.conv_gg = GCNConv(nhid, nhid)
        elif conv_type == 'GraphSAGE':
            self.conv = SAGEConv(nhid, nhid)
            if use_gg_conv:
                self.conv_gg = SAGEConv(nhid, nhid)
        elif conv_type == 'GAT':
            self.conv = GATConv(nhid, nhid, heads=4, concat=True)
            if use_gg_conv:
                self.conv_gg = GATConv(nhid, nhid, heads=4, concat=True)
        else:
            raise ValueError(f'Not implemented conv_type: {conv_type}')

        if use_gg_conv:
            self.G2 = G2(self.conv_gg, p, conv_type, activation=nn.ReLU())
        else:
            self.G2 = G2(self.conv, p, conv_type, activation=nn.ReLU())

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        n_nodes = x.size(0)

        x = F.dropout(x, self.drop_in, training=self.training)
        x = torch.relu(self.enc(x))

        for _ in range(self.nlayers):
            if self.conv_type == 'GAT':
                x_agg = F.elu(self.conv(x, edge_index)).view(n_nodes, -1, 4).mean(dim=-1)
            else:
                x_agg = torch.relu(self.conv(x, edge_index))

            tau = self.G2(x, edge_index)
            x = (1 - tau)*x + tau*x_agg

        x = F.dropout(x, self.drop, training=self.training)
        return self.dec(x)


class VanillaGNN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayers, conv_type='SAGE',
                 drop_in=0., drop=0.):
        super().__init__()
        self.conv_type = conv_type
        self.nlayers = nlayers
        self.drop_in = drop_in
        self.drop = drop

        self.enc = nn.Linear(nfeat, nhid)
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            if conv_type == 'GCN':
                self.convs.append(GCNConv(nhid, nhid))
            elif conv_type == 'GraphSAGE':
                self.convs.append(SAGEConv(nhid, nhid))
            elif conv_type == 'GAT':
                self.convs.append(GATConv(nhid, nhid, heads=4, concat=True))
            else:
                raise ValueError(f'Not implemented conv_type: {conv_type}')
        self.dec = nn.Linear(nhid, nclass)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        n_nodes = x.size(0)

        x = F.dropout(x, self.drop_in, training=self.training)
        x = torch.relu(self.enc(x))

        for conv in self.convs:
            if self.conv_type == 'GAT':
                x = F.elu(conv(x, edge_index)).view(n_nodes, -1, 4).mean(dim=-1)
            else:
                x = torch.relu(conv(x, edge_index))

        x = F.dropout(x, self.drop, training=self.training)
        return self.dec(x)


def compute_gamma_squash_global_mean(x, p=2.):
    global_mean = x.mean(dim=0, keepdim=True)  # shape [1,d]
    diffs = (x - global_mean).abs().pow(p).sum(dim=1)   # shape [N]
    diffs_tanh = torch.tanh(diffs)
    gamma_squash = 1.0 - diffs_tanh
    return gamma_squash.unsqueeze(-1)  # shape [N,1]


class DualGate_GNN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayers, 
                 conv_type='GCN', drop_in=0., drop=0., p=2.5):
        super().__init__()
        self.conv_type = conv_type
        self.nlayers = nlayers
        self.drop_in = drop_in
        self.drop = drop
        self.p = p

        self.enc = nn.Linear(nfeat, nhid, bias=True)
        self.skip_fc = nn.Linear(nfeat, nhid, bias=False)

        if conv_type == 'GCN':
            self.conv = GCNConv(nhid, nhid)
        elif conv_type == 'GraphSAGE':
            self.conv = SAGEConv(nhid, nhid)
        elif conv_type == 'GAT':
            self.conv = GATConv(nhid, nhid, heads=4, concat=True)
        else:
            raise ValueError(f'Not implemented conv_type: {conv_type}')

        self.dec = nn.Linear(nhid, nclass, bias=True)

    def forward(self, data):
        x0 = data.x
        edge_index = data.edge_index
        N = x0.size(0)

        # encode
        x = F.dropout(x0, self.drop_in, training=self.training)
        x = torch.relu(self.enc(x))
        x_skip = self.skip_fc(x0)  

        for _ in range(self.nlayers):
            # aggregator
            if self.conv_type == 'GAT':
                x_agg = F.elu(self.conv(x, edge_index)).view(N, -1, 4).mean(dim=-1)
            else:
                x_agg = torch.relu(self.conv(x, edge_index))

            # oversmoothing => local diffs
            gamma_smooth = self.compute_gamma_smooth(x, edge_index, p=2)
            # oversquash => global mean approach
            gamma_squash = compute_gamma_squash_global_mean(x, p=self.p)

            denom = 1. + gamma_smooth + gamma_squash
            A = 1.0 / denom
            B = gamma_smooth / denom
            C = gamma_squash / denom

            x = A*x + B*x_agg + C*x_skip

        x = F.dropout(x, self.drop, training=self.training)
        return self.dec(x)

    def compute_gamma_smooth(self, x, edge_index, p=2):
        """
        local diffs => oversmoothing gating
        """
        row, col = edge_index
        diffs = (x[row] - x[col]).abs().pow(p).sum(dim=-1)
        gamma = torch.zeros(x.size(0), device=x.device)
        gamma.index_add_(0, row, diffs)
        deg = torch.bincount(row).float() + 1e-10
        gamma = gamma/deg
        gamma = torch.tanh(gamma).unsqueeze(-1)
        return gamma


def build_model_wrapper(model_name, nfeat, nhid, nclass, nlayers,
                        drop_in=0., drop=0., use_g2_conv=True, p=2.):
    m = model_name.lower()
    if 'graphsage' in m:
        conv_type = 'GraphSAGE'
    elif 'gcn' in m:
        conv_type = 'GCN'
    elif 'gat' in m:
        conv_type = 'GAT'
    else:
        raise ValueError(f'Unknown aggregator in {model_name}')

    if m.startswith('g2'):
        net = G2_GNN(nfeat, nhid, nclass, nlayers, 
                     conv_type=conv_type, p=p, 
                     drop_in=drop_in, drop=drop, 
                     use_gg_conv=True)
    elif m.startswith('dual'):
        net = DualGate_GNN(nfeat, nhid, nclass, nlayers, 
                           conv_type=conv_type, 
                           drop_in=drop_in, drop=drop,
                           p=p)
    else:
        net = VanillaGNN(nfeat, nhid, nclass, nlayers, 
                         conv_type=conv_type,
                         drop_in=drop_in, drop=drop)
    return net


######################################################
# Training + Evaluate
######################################################
def run_experiment(args, split):
    """
    For a given dataset + model_name, do one split, train, return best test acc.
    """
    data = get_data(args.dataset, split).to(args.device)
    nclass = int(data.y.max().item()+1)
    model = build_model_wrapper(
        model_name=args.model,
        nfeat=data.num_node_features,
        nhid=args.nhid,
        nclass=nclass,
        nlayers=args.nlayers,
        drop_in=args.drop_in,
        drop=args.drop,
        use_g2_conv=True,
        p=args.G2_exp
    ).to(args.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def test_eval():
        model.eval()
        out = model(data)
        logits = out
        loss_train = criterion(logits[data.train_mask], data.y[data.train_mask])
        loss_val   = criterion(logits[data.val_mask],   data.y[data.val_mask])
        loss_test  = criterion(logits[data.test_mask],  data.y[data.test_mask])

        pred_train = logits[data.train_mask].max(dim=1)[1]
        pred_val   = logits[data.val_mask].max(dim=1)[1]
        pred_test  = logits[data.test_mask].max(dim=1)[1]

        acc_train = float((pred_train == data.y[data.train_mask]).sum().item())/ data.train_mask.sum().item()
        acc_val   = float((pred_val   == data.y[data.val_mask]).sum().item())  / data.val_mask.sum().item()
        acc_test  = float((pred_test  == data.y[data.test_mask]).sum().item()) / data.test_mask.sum().item()

        return (acc_train, acc_val, acc_test), (loss_train.item(), loss_val.item(), loss_test.item())

    best_val_acc = 0.
    best_test_acc = 0.
    best_val_loss = 9999.
    bad_counter = 0

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss_train = criterion(out[data.train_mask], data.y[data.train_mask])
        loss_train.backward()
        optimizer.step()

        (acc_tr, acc_val, acc_test), (lt, lv, ltst) = test_eval()
        if args.use_val_acc:
            if acc_val > best_val_acc:
                best_val_acc = acc_val
                best_test_acc = acc_test
                bad_counter = 0
            else:
                bad_counter += 1
        else:
            if lv < best_val_loss:
                best_val_loss = lv
                best_test_acc = acc_test
                bad_counter = 0
            else:
                bad_counter += 1

        if bad_counter >= args.patience:
            break

    return best_test_acc


######################################################
# Main
######################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Heterophily test (9 variants) with global-mean oversquash gating in Dual.')
    parser.add_argument('--dataset', type=str, default='chameleon',
                        help='One of: cornell, texas, wisconsin, chameleon, squirrel, film.')
    parser.add_argument('--model', type=str, default='dual-gat',
                        help='Which aggregator: gcn,gat,graphsage, g2-gcn, g2-gat, g2-graphsage, dual-gcn, dual-gat, dual-graphsage.')
    parser.add_argument('--nhid', type=int, default=256)
    parser.add_argument('--nlayers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--patience', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--drop_in', type=float, default=0.5)
    parser.add_argument('--drop', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--G2_exp', type=float, default=2.5)
    parser.add_argument('--use_val_acc', action='store_true')
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()
    args.device = torch.device(args.device)

    n_splits = 10
    best_list = []
    for split in range(n_splits):
        acc_test = run_experiment(args, split)
        best_list.append(acc_test)
        print(f"Split {split}: test_acc={acc_test:.4f}")

    best_list = np.array(best_list)
    mean_acc = np.mean(best_list)
    std_acc  = np.std(best_list)
    print(f"Final test results on {args.dataset}, model={args.model} => mean: {mean_acc:.4f} ± {std_acc:.4f}")
