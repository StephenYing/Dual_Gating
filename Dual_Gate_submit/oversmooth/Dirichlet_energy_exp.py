import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter

import networkx as nx
import matplotlib.pyplot as plt

try:
    from torch_geometric.utils import from_networkx
    from torch_geometric.nn import GCNConv, GATConv
except ImportError:
    raise ImportError("Please install torch_geometric and torch_scatter.")


def build_2d_grid_graph(grid_size=10):
    G = nx.grid_2d_graph(grid_size, grid_size)
    G = nx.convert_node_labels_to_integers(G) 
    data = from_networkx(G)
    return data

def dirichlet_energy(x, edge_index):
    row, col = edge_index
    diff = x[row] - x[col]  # [E, d]
    diff_sq = (diff**2).sum(dim=-1)  # [E]
    return 0.5 * diff_sq.sum().item()


class G2(nn.Module):

    def __init__(self, conv, p=2., conv_type='GCN', activation=nn.ReLU()):
        super().__init__()
        self.conv = conv
        self.p = p
        self.activation = activation
        self.conv_type = conv_type

    def forward(self, X, edge_index):
        n_nodes = X.size(0)
        if self.conv_type == 'GAT':
            X_agg = F.elu(self.conv(X, edge_index)).view(n_nodes, -1, 4).mean(dim=-1)
        else:  
            X_agg = self.activation(self.conv(X, edge_index))


        row, col = edge_index
        diff = (torch.abs(X[row] - X[col]) ** self.p).sum(dim=-1)  
        gg = scatter(diff, row, dim=0, dim_size=n_nodes, reduce='mean')
        gg = torch.tanh(gg)  
        return gg


class G2_GNN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayers,
                 conv_type='GCN', p=2., drop_in=0., drop=0., use_gg_conv=True):
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
        elif conv_type == 'GAT':
            self.conv = GATConv(nhid, nhid, heads=4, concat=True)
            if use_gg_conv:
                self.conv_gg = GATConv(nhid, nhid, heads=4, concat=True)
        else:
            raise NotImplementedError("conv_type must be 'GCN' or 'GAT'")

        if use_gg_conv:
            self.G2 = G2(self.conv_gg, p, conv_type, activation=nn.ReLU())
        else:
            self.G2 = G2(self.conv, p, conv_type, activation=nn.ReLU())

    def forward(self, data, return_all=False):
        X = data.x
        edge_index = data.edge_index
        n_nodes = X.size(0)

        X = F.dropout(X, self.drop_in, training=self.training)
        X = torch.relu(self.enc(X))

        # For Dirichlet experiment
        all_states = [X.clone()]

        for _ in range(self.nlayers):
            # aggregator
            if self.conv_type == 'GAT':
                X_agg = F.elu(self.conv(X, edge_index)).view(n_nodes, -1, 4).mean(dim=-1)
            else:
                X_agg = torch.relu(self.conv(X, edge_index))
            # gating
            tau = self.G2(X, edge_index).unsqueeze(-1)  

            # update
            X = (1 - tau)*X + tau*X_agg
            all_states.append(X.clone())

        X = F.dropout(X, self.drop, training=self.training)
        out = self.dec(X)

        if return_all:
            return out, all_states
        else:
            return out


class plain_GNN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayers, conv_type='GCN', drop_in=0., drop=0.):
        super().__init__()
        self.conv_type = conv_type
        self.drop_in = drop_in
        self.drop = drop
        self.nlayers = nlayers

        self.enc = nn.Linear(nfeat, nhid)
        if conv_type == 'GCN':
            self.convs = nn.ModuleList([GCNConv(nhid, nhid) for _ in range(nlayers)])
        elif conv_type == 'GAT':
            self.convs = nn.ModuleList([GATConv(nhid, nhid, heads=4, concat=True) for _ in range(nlayers)])
        else:
            raise NotImplementedError("Use 'GCN' or 'GAT'")

        self.dec = nn.Linear(nhid, nclass)

    def forward(self, data, return_all=False):
        X = data.x
        edge_index = data.edge_index
        n_nodes = X.size(0)

        X = F.dropout(X, self.drop_in, training=self.training)
        X = torch.relu(self.enc(X))

        all_states = [X.clone()]

        for conv in self.convs:
            if self.conv_type == 'GAT':
                X = F.elu(conv(X, edge_index)).view(n_nodes, -1, 4).mean(dim=-1)
            else:
                X = torch.relu(conv(X, edge_index))
            all_states.append(X.clone())

        X = F.dropout(X, self.drop, training=self.training)
        X = self.dec(X)

        if return_all:
            return X, all_states
        else:
            return X


class DualGating_GNN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayers,
                 conv_type='GCN', p=2.,
                 drop_in=0., drop=0., use_gg_conv=True):
        super().__init__()
        self.conv_type = conv_type
        self.nlayers = nlayers
        self.drop_in = drop_in
        self.drop = drop

        self.enc = nn.Linear(nfeat, nhid)
        self.dec = nn.Linear(nhid, nclass)

        if conv_type == 'GCN':
            self.conv = GCNConv(nhid, nhid)
            if use_gg_conv:
                self.conv_gg_smooth = GCNConv(nhid, nhid)
                self.conv_gg_squash = GCNConv(nhid, nhid)
        elif conv_type == 'GAT':
            self.conv = GATConv(nhid, nhid, heads=4, concat=True)
            if use_gg_conv:
                self.conv_gg_smooth = GATConv(nhid, nhid, heads=4, concat=True)
                self.conv_gg_squash = GATConv(nhid, nhid, heads=4, concat=True)
        else:
            raise NotImplementedError("conv_type must be 'GCN' or 'GAT'")

        if use_gg_conv:
            self.G2_smooth = G2(self.conv_gg_smooth, p=p, conv_type=conv_type, activation=nn.ReLU())
            self.G2_squash = G2(self.conv_gg_squash, p=p, conv_type=conv_type, activation=nn.ReLU())
        else:
            self.G2_smooth = None
            self.G2_squash = None

        self.W_skip = nn.Linear(nhid, nhid, bias=False)

    def forward(self, data, return_all=False):
        X0 = data.x
        edge_index = data.edge_index
        n_nodes = X0.size(0)

        X = F.dropout(X0, self.drop_in, training=self.training)
        X = torch.relu(self.enc(X))
        X_init = X.clone()  

        all_states = [X.clone()]

        for _ in range(self.nlayers):
            # aggregator
            if self.conv_type == 'GAT':
                X_agg = F.elu(self.conv(X, edge_index)).view(n_nodes, -1, 4).mean(dim=-1)
            else:
                X_agg = torch.relu(self.conv(X, edge_index))

            # gamma_smooth, gamma_squash
            if self.G2_smooth is not None:
                gamma_smooth = self.G2_smooth(X, edge_index)
                gamma_squash = self.G2_squash(X, edge_index)
            else:
                gamma_smooth = torch.zeros(n_nodes, device=X.device)
                gamma_squash = torch.zeros(n_nodes, device=X.device)

            gamma_smooth = gamma_smooth.unsqueeze(-1)
            gamma_squash = gamma_squash.unsqueeze(-1)

            denom = 1.0 + gamma_smooth + gamma_squash
            A = 1.0 / denom
            B = gamma_smooth / denom
            C = gamma_squash / denom

            skip_val = self.W_skip(X_init)

            # combine
            X = A * X + B * X_agg + C * skip_val
            all_states.append(X.clone())

        X = F.dropout(X, self.drop, training=self.training)
        X = self.dec(X)

        if return_all:
            return X, all_states
        else:
            return X



def run_dirichlet_experiment_six_lines(grid_size=10, nlayers=50):

    # Build grid data
    data = build_2d_grid_graph(grid_size)
    N = data.num_nodes
    in_dim = 4
    data.x = torch.rand(N, in_dim)  # random [0,1]

    nhid = 4
    nclass = 2

    # Build the six models
    model_gcn = plain_GNN(nfeat=in_dim, nhid=nhid, nclass=nclass,
                          nlayers=nlayers, conv_type='GCN')
    model_gat = plain_GNN(nfeat=in_dim, nhid=nhid, nclass=nclass,
                          nlayers=nlayers, conv_type='GAT')
    model_g2_gcn = G2_GNN(nfeat=in_dim, nhid=nhid, nclass=nclass,
                          nlayers=nlayers, conv_type='GCN', use_gg_conv=True)
    model_g2_gat = G2_GNN(nfeat=in_dim, nhid=nhid, nclass=nclass,
                          nlayers=nlayers, conv_type='GAT', use_gg_conv=True)
    model_dual_gcn = DualGating_GNN(nfeat=in_dim, nhid=nhid, nclass=nclass,
                                    nlayers=nlayers, conv_type='GCN', use_gg_conv=True)
    model_dual_gat = DualGating_GNN(nfeat=in_dim, nhid=nhid, nclass=nclass,
                                    nlayers=nlayers, conv_type='GAT', use_gg_conv=True)

    models = [
        ("GCN",              model_gcn),
        ("GAT",              model_gat),
        ("G^2-GCN",          model_g2_gcn),
        ("G^2-GAT",          model_g2_gat),
        ("Dual-Gating-GCN", model_dual_gcn),
        ("Dual-Gating-GAT", model_dual_gat),
    ]


    results = {}
    for label, net in models:
        net.eval()
        with torch.no_grad():
            _, all_states = net(data, return_all=True)
        energies = []
        for X_state in all_states:
            e = dirichlet_energy(X_state, data.edge_index)
            energies.append(e)
        results[label] = energies

    plt.figure(figsize=(8,6))

    plot_styles = {
        "GCN": {
            "color": "cornflowerblue",
            "marker": "o",
            "linestyle": "-",
            "label": "GCN"
        },
        "GAT": {
            "color": "darkorange",
            "marker": "s",
            "linestyle": "-",
            "label": "GAT"
        },
        "G^2-GCN": {
            "color": "crimson",
            "marker": "^",
            "linestyle": "--",
            "label": r"G$^2$-GCN"
        },
        "G^2-GAT": {
            "color": "darkviolet",
            "marker": "v",
            "linestyle": "--",
            "label": r"G$^2$-GAT"
        },
        "Dual-Gating-GCN": {
            "color": "seagreen",
            "marker": "D",
            "linestyle": "-.",
            "label": "Dual-Gating-GCN"
        },
        "Dual-Gating-GAT": {
            "color": "saddlebrown",
            "marker": "X",
            "linestyle": "-.",
            "label": "Dual-Gating-GAT"
        },
    }

    for label, energies in results.items():
        style = plot_styles[label]
        plt.plot(
            range(len(energies)),
            energies,
            linewidth=2.0,
            markersize=5,
            **style
        )

    plt.yscale('log')
    plt.xlabel("Layer", fontsize=12)
    plt.ylim([1e-26, 1e2])
    plt.ylabel("Dirichlet Energy", fontsize=12)
    plt.title(f"Dirichlet Energy on {grid_size}x{grid_size} Grid", fontsize=13)

    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    run_dirichlet_experiment_six_lines(grid_size=10, nlayers=50)
