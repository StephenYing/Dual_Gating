###############################################
# acc_heterophily_all_in_csv.py
###############################################
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import csv

from torch_scatter import scatter
from torch_geometric.nn import (GCNConv, GATConv, SAGEConv)
from torch_geometric.datasets import WebKB, WikipediaNetwork, Actor
from torch_geometric.utils import to_undirected

import argparse

######################################################
# 1) Data Handling
######################################################
def get_data(name, split=0):
    """
    Loads a heterophilic dataset (Cornell, Texas, Wisconsin, Chameleon,
    Squirrel, Film) from local path, along with a particular 'split' (0..9).
    Binds train/val/test masks to data.
    """
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

    # geometry splits:
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

    # edges undirected
    data.edge_index = to_undirected(data.edge_index)
    return data


######################################################
# 2) Minimal Model Definitions (Vanilla, G^2, Dual)
######################################################
# For brevity, we'll define simpler versions here or you can import from your existing script.

class VanillaGNN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayers, conv_type='GCN',
                 drop_in=0., drop=0.):
        super().__init__()
        self.conv_type = conv_type
        self.nlayers = nlayers
        self.drop_in = drop_in
        self.drop = drop
        self.enc = nn.Linear(nfeat, nhid)
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            if conv_type=='GCN':
                self.convs.append(GCNConv(nhid, nhid))
            elif conv_type=='GAT':
                self.convs.append(GATConv(nhid, nhid, heads=4, concat=True))
            elif conv_type=='GraphSAGE':
                self.convs.append(SAGEConv(nhid, nhid))
            else:
                raise ValueError(f'Unknown conv_type: {conv_type}')
        self.dec = nn.Linear(nhid, nclass)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        n_nodes = x.size(0)
        x = F.dropout(x, self.drop_in, training=self.training)
        x = F.relu(self.enc(x))
        for conv in self.convs:
            if self.conv_type=='GAT':
                x = F.elu(conv(x, edge_index)).view(n_nodes, -1, 4).mean(dim=-1)
            else:
                x = F.relu(conv(x, edge_index))
        x = F.dropout(x, self.drop, training=self.training)
        return self.dec(x)


class G2(nn.Module):
    def __init__(self, conv, p=2., conv_type='GCN'):
        super().__init__()
        self.conv=conv
        self.p=p
        self.conv_type=conv_type

    def forward(self, X, edge_index):
        n_nodes = X.size(0)
        if self.conv_type=='GAT':
            X = F.elu(self.conv(X, edge_index)).view(n_nodes, -1, 4).mean(dim=-1)
        else:
            X = F.relu(self.conv(X, edge_index))
        row, col = edge_index
        diffs = (X[row]-X[col]).abs().pow(self.p).sum(dim=-1)
        gg = scatter(diffs, row, dim=0, reduce='mean')
        return torch.tanh(gg)


class G2_GNN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayers, conv_type='GCN',
                 p=2., drop_in=0., drop=0., use_gg_conv=True):
        super().__init__()
        self.enc = nn.Linear(nfeat, nhid)
        self.dec = nn.Linear(nhid, nclass)
        self.drop_in=drop_in
        self.drop=drop
        self.nlayers=nlayers
        self.conv_type=conv_type

        if conv_type=='GCN':
            self.conv=GCNConv(nhid, nhid)
            if use_gg_conv:
                self.conv_gg=GCNConv(nhid, nhid)
        elif conv_type=='GAT':
            self.conv=GATConv(nhid, nhid, heads=4, concat=True)
            if use_gg_conv:
                self.conv_gg=GATConv(nhid, nhid, heads=4, concat=True)
        elif conv_type=='GraphSAGE':
            self.conv=SAGEConv(nhid,nhid)
            if use_gg_conv:
                self.conv_gg=SAGEConv(nhid,nhid)
        else:
            raise ValueError(f'Unknown conv {conv_type}')

        if use_gg_conv:
            self.G2=G2(self.conv_gg, p, conv_type=conv_type)
        else:
            self.G2=G2(self.conv, p, conv_type=conv_type)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        n_nodes = x.size(0)
        x = F.dropout(x, self.drop_in, training=self.training)
        x = F.relu(self.enc(x))

        for _ in range(self.nlayers):
            if self.conv_type=='GAT':
                x_agg=F.elu(self.conv(x, edge_index)).view(n_nodes, -1, 4).mean(dim=-1)
            else:
                x_agg=F.relu(self.conv(x, edge_index))
            tau=self.G2(x, edge_index)
            # shape [N], expand to [N,1] if needed
            tau = tau.unsqueeze(-1)
            x = (1 - tau)*x + tau*x_agg

        x=F.dropout(x, self.drop, training=self.training)
        return self.dec(x)


class DualGate_GNN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayers, conv_type='GCN',
                 drop_in=0., drop=0., p=2.5):
        super().__init__()
        self.conv_type=conv_type
        self.nlayers=nlayers
        self.drop_in=drop_in
        self.drop=drop
        self.p=p

        self.enc=nn.Linear(nfeat, nhid)
        self.skip_fc=nn.Linear(nfeat, nhid, bias=False)

        if conv_type=='GCN':
            self.conv=GCNConv(nhid, nhid)
        elif conv_type=='GraphSAGE':
            self.conv=SAGEConv(nhid, nhid)
        elif conv_type=='GAT':
            self.conv=GATConv(nhid, nhid, heads=4, concat=True)
        else:
            raise ValueError(f'Not implemented conv_type: {conv_type}')

        self.dec=nn.Linear(nhid, nclass)

    def forward(self, data):
        x0=data.x
        edge_index=data.edge_index
        N=x0.size(0)

        x=F.dropout(x0, self.drop_in, training=self.training)
        x=torch.relu(self.enc(x))

        x_skip=self.skip_fc(x0)

        for _ in range(self.nlayers):
            if self.conv_type=='GAT':
                x_agg=F.elu(self.conv(x, edge_index)).view(N, -1, 4).mean(dim=-1)
            else:
                x_agg=F.relu(self.conv(x, edge_index))

            gamma_smooth=self.compute_gamma_smooth(x, edge_index)
            gamma_squash=self.compute_gamma_squash_global(x, p=self.p)

            denom=1. + gamma_smooth + gamma_squash
            A=1.0/denom
            B=gamma_smooth/denom
            C=gamma_squash/denom

            x = A*x + B*x_agg + C*x_skip

        x=F.dropout(x, self.drop, training=self.training)
        return self.dec(x)

    def compute_gamma_smooth(self, x, edge_index, p=2):
        row,col = edge_index
        diffs=(x[row]-x[col]).abs().pow(p).sum(dim=-1)
        gamma=torch.zeros(x.size(0), device=x.device)
        gamma.index_add_(0, row, diffs)
        deg=torch.bincount(row).float()+1e-10
        gamma=gamma/deg
        gamma=torch.tanh(gamma).unsqueeze(-1)
        return gamma

    def compute_gamma_squash_global(self, x, p=2.):
        # gamma_squash(i)=1 - tanh(||x_i - mean||^p)
        global_mean = x.mean(dim=0, keepdim=True)
        d = (x - global_mean).abs().pow(p).sum(dim=-1)
        # shape [N]
        d_tanh=torch.tanh(d)
        gamma_squash=1. - d_tanh
        return gamma_squash.unsqueeze(-1)


######################################################
# 3) build_model_wrapper
######################################################
def build_model_wrapper(model_name, nfeat, nhid, nclass, nlayers,
                        drop_in=0., drop=0., use_g2_conv=True, p=2.):
    m = model_name.lower()
    if 'graphsage' in m:
        conv_type='GraphSAGE'
    elif 'gcn' in m:
        conv_type='GCN'
    elif 'gat' in m:
        conv_type='GAT'
    else:
        raise ValueError(f'Unknown aggregator in {model_name}')

    if m.startswith('g2'):
        net = G2_GNN(
            nfeat, nhid, nclass, nlayers, 
            conv_type=conv_type, p=p,
            drop_in=drop_in, drop=drop, 
            use_gg_conv=True
        )
    elif m.startswith('dual'):
        net=DualGate_GNN(
            nfeat, nhid, nclass, nlayers,
            conv_type=conv_type, drop_in=drop_in, drop=drop, p=p
        )
    else:
        net=VanillaGNN(
            nfeat, nhid, nclass, nlayers,
            conv_type=conv_type, drop_in=drop_in, drop=drop
        )
    return net


######################################################
# 4) Single training function
######################################################
def run_experiment(args, dataset_name, model_name, split):
    data=get_data(dataset_name, split).to(args.device)
    nclass=int(data.y.max().item()+1)
    net=build_model_wrapper(model_name, 
                            data.num_node_features, args.nhid,
                            nclass, args.nlayers,
                            drop_in=args.drop_in, drop=args.drop,
                            use_g2_conv=True, p=args.G2_exp
    ).to(args.device)

    crit=nn.CrossEntropyLoss()
    optimr=optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def test_eval():
        net.eval()
        with torch.no_grad():
            out=net(data)
        loss_train=crit(out[data.train_mask], data.y[data.train_mask])
        loss_val=crit(out[data.val_mask], data.y[data.val_mask])
        loss_test=crit(out[data.test_mask], data.y[data.test_mask])

        pred_train=out[data.train_mask].max(dim=1)[1]
        pred_val=out[data.val_mask].max(dim=1)[1]
        pred_test=out[data.test_mask].max(dim=1)[1]

        acc_train=float((pred_train==data.y[data.train_mask]).sum().item())/data.train_mask.sum().item()
        acc_val=float((pred_val==data.y[data.val_mask]).sum().item())/data.val_mask.sum().item()
        acc_test=float((pred_test==data.y[data.test_mask]).sum().item())/data.test_mask.sum().item()
        return (acc_train, acc_val, acc_test),(loss_train.item(),loss_val.item(),loss_test.item())

    best_val_acc=0.
    best_val_loss=1e9
    best_test_acc=0.
    bad_counter=0

    for epoch in range(args.epochs):
        net.train()
        optimr.zero_grad()
        out=net(data)
        loss_train=crit(out[data.train_mask], data.y[data.train_mask])
        loss_train.backward()
        optimr.step()

        (acc_tr,acc_val,acc_test),(lt,lv,ltst)=test_eval()
        if args.use_val_acc:
            if acc_val>best_val_acc:
                best_val_acc=acc_val
                best_test_acc=acc_test
                bad_counter=0
            else:
                bad_counter+=1
        else:
            if lv<best_val_loss:
                best_val_loss=lv
                best_test_acc=acc_test
                bad_counter=0
            else:
                bad_counter+=1

        if bad_counter>=args.patience:
            break

    return best_test_acc


######################################################
# 5) Main => run all combos & save CSV
######################################################
if __name__=='__main__':
    parser=argparse.ArgumentParser()
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
    parser.add_argument('--output_csv', type=str, default='acc_table.csv',
                        help='CSV file to store results')
    args=parser.parse_args()
    args.device=torch.device(args.device)

    DATASETS=['cornell','texas','wisconsin','chameleon','squirrel','film']
    MODELS=[
        'gcn','gat','graphsage',
        'g2-gcn','g2-gat','g2-graphsage',
        'dual-gcn','dual-gat','dual-graphsage'
    ]

    results=[]
    n_splits=10

    for ds in DATASETS:
        for md in MODELS:
            test_accs=[]
            for sp in range(n_splits):
                acc=run_experiment(args, ds, md, sp)
                test_accs.append(acc)
                print(f"[{ds} | {md}] Split={sp}, TestAcc={acc:.4f}")
            arr=np.array(test_accs)
            mean_val=arr.mean()
            std_val=arr.std()
            print(f"=> Final {ds}, {md} => {mean_val:.4f} ± {std_val:.4f}")

            # store in a row => dataset,model,"0.8118 ± 0.0576"
            row=[ds, md, f"{mean_val:.4f} ± {std_val:.4f}"]
            results.append(row)

    # write CSV
    import csv
    with open(args.output_csv, 'w', newline='') as f:
        writer=csv.writer(f)
        writer.writerow(["dataset","model","acc ± std"])
        for row in results:
            writer.writerow(row)

    print(f"Saved results to {args.output_csv}")
