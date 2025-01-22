#######################################
# factory.py
#######################################
import torch
import torch.nn as nn
import torch.nn.functional as F


from torch_geometric.nn import GIN, GCN, GAT, GraphSAGE, GCNConv, GATConv

from data.ring_transfer import (
    generate_tree_transfer_graph_dataset,
    generate_ring_transfer_graph_dataset,
    generate_lollipop_transfer_graph_dataset
)



def build_two_hop_edges(edge_index, num_nodes):
    """
    Builds a 2-hop adjacency from the original 1-hop edge_index [2, E].
    For each node i, gather neighbors-of-neighbors => (i, n2).
    """
    device = edge_index.device
    row, col = edge_index

    neighbors = [[] for _ in range(num_nodes)]
    for r, c in zip(row.tolist(), col.tolist()):
        neighbors[r].append(c)

    two_hop_row = []
    two_hop_col = []
    for i in range(num_nodes):
        n2_set = set()
        for n1 in neighbors[i]:
            n2_set.update(neighbors[n1])
        if i in n2_set:
            n2_set.remove(i)
        for n2 in n2_set:
            two_hop_row.append(i)
            two_hop_col.append(n2)

    row2 = torch.tensor(two_hop_row, device=device, dtype=torch.long)
    col2 = torch.tensor(two_hop_col, device=device, dtype=torch.long)
    edge_index_2hop = torch.stack([row2, col2], dim=0)
    return edge_index_2hop


########################################################
# G2 Single-Gate Classes (GCN / GAT)
########################################################
class G2GCNModel(nn.Module):
    """
    Multi-layer GCN with G2 gating logic.
    Called as model(x, edge_index).
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, norm=None):
        super().__init__()
        self.num_layers = num_layers

        self.input_fc = nn.Linear(in_channels, hidden_channels, bias=False)
        self.convs = nn.ModuleList([
            GCNConv(hidden_channels, hidden_channels, normalize=norm)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(hidden_channels, out_channels, bias=True)

    def forward(self, x, edge_index):
        x = self.input_fc(x)
        for conv in self.convs:
            x_new = F.relu(conv(x, edge_index))
            tau = self.compute_g2_tau(x, x_new, edge_index)
            x = (1 - tau)*x + tau*x_new
        return self.fc(x)

    def compute_g2_tau(self, old_x, new_x, edge_index, p=2):
        row, col = edge_index
        diffs = (new_x[row] - new_x[col]).abs().pow(p).sum(dim=-1)
        num_nodes = old_x.size(0)
        tau_vals = torch.zeros(num_nodes, device=old_x.device)
        tau_vals.index_add_(0, row, diffs)
        deg = torch.bincount(row).float() + 1e-10
        tau_vals = tau_vals / deg
        tau_vals = torch.tanh(tau_vals).unsqueeze(-1)
        return tau_vals


class G2GATModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, norm=None):
        super().__init__()
        self.num_layers = num_layers

        self.input_fc = nn.Linear(in_channels, hidden_channels, bias=False)
        self.convs = nn.ModuleList([
            GATConv(hidden_channels, hidden_channels, heads=1, concat=False)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(hidden_channels, out_channels, bias=True)

    def forward(self, x, edge_index):
        x = self.input_fc(x)
        for conv in self.convs:
            x_new = F.elu(conv(x, edge_index))
            tau = self.compute_g2_tau(x, x_new, edge_index)
            x = (1 - tau)*x + tau*x_new
        return self.fc(x)

    def compute_g2_tau(self, old_x, new_x, edge_index, p=2):
        row, col = edge_index
        diffs = (new_x[row] - new_x[col]).abs().pow(p).sum(dim=-1)
        num_nodes = old_x.size(0)
        tau_vals = torch.zeros(num_nodes, device=old_x.device)
        tau_vals.index_add_(0, row, diffs)
        deg = torch.bincount(row).float() + 1e-10
        tau_vals = tau_vals / deg
        tau_vals = torch.tanh(tau_vals).unsqueeze(-1)
        return tau_vals


########################################################
# Dual-Gate Classes (GCN / GAT) - 1-hop only
########################################################
class DualGate_GCNModel(nn.Module):
    """
    Called as model(x, edge_index, x0).
    Oversmoothing & oversquash from local diffs (1-hop).
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, norm=None):
        super().__init__()
        self.num_layers = num_layers

        self.input_fc = nn.Linear(in_channels, hidden_channels, bias=False)
        self.skip_fc  = nn.Linear(hidden_channels, hidden_channels, bias=False)

        self.convs = nn.ModuleList([
            GCNConv(hidden_channels, hidden_channels, normalize=norm)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(hidden_channels, out_channels, bias=True)

    def forward(self, x, edge_index, x0):
        x  = self.input_fc(x)
        x0 = self.input_fc(x0)

        for conv in self.convs:
            x_agg = F.relu(conv(x, edge_index))
            Gamma_smooth = self.compute_gamma_local_diff(x_agg, edge_index)
            Gamma_squash = self.compute_gamma_local_diff(x_agg, edge_index)

            A, B, C = self.compute_abc(x, x_agg, x0, Gamma_smooth, Gamma_squash)
            x_skip = self.skip_fc(x0)
            x = A*x + B*x_agg + C*x_skip

        return self.fc(x)

    def compute_gamma_local_diff(self, x_agg, edge_index, p=2):
        row, col = edge_index
        diffs = (x_agg[row] - x_agg[col]).abs().pow(p).sum(dim=-1)
        num_nodes = x_agg.size(0)
        gamma = torch.zeros(num_nodes, device=x_agg.device)
        gamma.index_add_(0, row, diffs)
        deg = torch.bincount(row).float() + 1e-10
        gamma = gamma / deg
        gamma = torch.tanh(gamma)
        gamma = gamma.unsqueeze(1).expand_as(x_agg)
        return gamma

    def compute_abc(self, x, x_agg, x0, Gamma_smooth, Gamma_squash):
        denom = 1.0 + Gamma_smooth + Gamma_squash
        A = 1.0 / denom
        B = Gamma_smooth / denom
        C = Gamma_squash / denom
        return A, B, C


class DualGate_GATModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.num_layers = num_layers

        self.input_fc = nn.Linear(in_channels, hidden_channels, bias=False)
        self.skip_fc  = nn.Linear(hidden_channels, hidden_channels, bias=False)

        self.convs = nn.ModuleList([
            GATConv(hidden_channels, hidden_channels, heads=1, concat=False)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(hidden_channels, out_channels, bias=True)

    def forward(self, x, edge_index, x0):
        x  = self.input_fc(x)
        x0 = self.input_fc(x0)

        for conv in self.convs:
            x_agg = F.elu(conv(x, edge_index))
            Gamma_smooth = self.compute_gamma_local_diff(x_agg, edge_index)
            Gamma_squash = self.compute_gamma_local_diff(x_agg, edge_index)

            A, B, C = self.compute_abc(x, x_agg, x0, Gamma_smooth, Gamma_squash)
            x_skip = self.skip_fc(x0)
            x = A*x + B*x_agg + C*x_skip

        return self.fc(x)

    def compute_gamma_local_diff(self, x_agg, edge_index, p=2):
        row, col = edge_index
        diffs = (x_agg[row] - x_agg[col]).abs().pow(p).sum(dim=-1)
        num_nodes = x_agg.size(0)
        gamma = torch.zeros(num_nodes, device=x_agg.device)
        gamma.index_add_(0, row, diffs)
        deg = torch.bincount(row).float() + 1e-10
        gamma = gamma / deg
        gamma = torch.tanh(gamma)
        gamma = gamma.unsqueeze(1).expand_as(x_agg)
        return gamma

    def compute_abc(self, x, x_agg, x0, Gamma_smooth, Gamma_squash):
        denom = 1.0 + Gamma_smooth + Gamma_squash
        A = 1.0 / denom
        B = Gamma_smooth / denom
        C = Gamma_squash / denom
        return A, B, C


########################################################
# Dual-Hop Classes (GCN / GAT) => 2-hop for oversquash
########################################################
class DualHopGCNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, norm=None):
        super().__init__()
        self.num_layers = num_layers

        self.input_fc = nn.Linear(in_channels, hidden_channels, bias=False)
        self.skip_fc  = nn.Linear(hidden_channels, hidden_channels, bias=False)

        self.convs = nn.ModuleList([
            GCNConv(hidden_channels, hidden_channels, normalize=norm)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(hidden_channels, out_channels, bias=True)

    def forward(self, x, edge_index, x0):
        device = x.device
        num_nodes = x.size(0)

        x  = self.input_fc(x)
        x0 = self.input_fc(x0)

        edge_index_2hop = build_two_hop_edges(edge_index, num_nodes).to(device)

        for conv in self.convs:
            x_agg = F.relu(conv(x, edge_index))

            Gamma_smooth = self.compute_gamma_smooth(x_agg, edge_index)
            Gamma_squash = self.compute_gamma_squash(x_agg, edge_index_2hop)

            A, B, C = self.compute_abc(x, x_agg, x0, Gamma_smooth, Gamma_squash)
            x_skip = self.skip_fc(x0)

            x = A*x + B*x_agg + C*x_skip

        return self.fc(x)

    def compute_gamma_smooth(self, x_agg, edge_index, p=2):
        row, col = edge_index
        diffs = (x_agg[row] - x_agg[col]).abs().pow(p).sum(dim=-1)
        num_nodes = x_agg.size(0)
        gamma = torch.zeros(num_nodes, device=x_agg.device)
        gamma.index_add_(0, row, diffs)
        deg = torch.bincount(row).float() + 1e-10
        gamma = gamma / deg
        gamma = torch.tanh(gamma)
        gamma = gamma.unsqueeze(1).expand_as(x_agg)
        return gamma

    def compute_gamma_squash(self, x_agg, edge_index_2hop, p=2):
        row2, col2 = edge_index_2hop
        diffs2 = (x_agg[row2] - x_agg[col2]).abs().pow(p).sum(dim=-1)
        num_nodes = x_agg.size(0)
        gamma2 = torch.zeros(num_nodes, device=x_agg.device)
        gamma2.index_add_(0, row2, diffs2)
        deg2 = torch.bincount(row2).float() + 1e-10
        gamma2 = gamma2 / deg2
        gamma2 = torch.tanh(gamma2)
        gamma2 = gamma2.unsqueeze(1).expand_as(x_agg)
        return gamma2

    def compute_abc(self, x, x_agg, x0, Gamma_smooth, Gamma_squash):
        denom = 1.0 + Gamma_smooth + Gamma_squash
        A = 1.0 / denom
        B = Gamma_smooth / denom
        C = Gamma_squash / denom
        return A, B, C


class DualHopGATModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.num_layers = num_layers

        self.input_fc = nn.Linear(in_channels, hidden_channels, bias=False)
        self.skip_fc  = nn.Linear(hidden_channels, hidden_channels, bias=False)

        self.convs = nn.ModuleList([
            GATConv(hidden_channels, hidden_channels, heads=1, concat=False)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(hidden_channels, out_channels, bias=True)

    def forward(self, x, edge_index, x0):
        device = x.device
        num_nodes = x.size(0)

        x  = self.input_fc(x)
        x0 = self.input_fc(x0)

        edge_index_2hop = build_two_hop_edges(edge_index, num_nodes).to(device)

        for conv in self.convs:
            x_agg = F.elu(conv(x, edge_index))

            Gamma_smooth = self.compute_gamma_smooth(x_agg, edge_index)
            Gamma_squash = self.compute_gamma_squash(x_agg, edge_index_2hop)

            A, B, C = self.compute_abc(x, x_agg, x0, Gamma_smooth, Gamma_squash)
            x_skip = self.skip_fc(x0)
            x = A*x + B*x_agg + C*x_skip

        return self.fc(x)

    def compute_gamma_smooth(self, x_agg, edge_index, p=2):
        row, col = edge_index
        diffs = (x_agg[row] - x_agg[col]).abs().pow(p).sum(dim=-1)
        num_nodes = x_agg.size(0)
        gamma = torch.zeros(num_nodes, device=x_agg.device)
        gamma.index_add_(0, row, diffs)
        deg = torch.bincount(row).float() + 1e-10
        gamma = gamma / deg
        gamma = torch.tanh(gamma)
        gamma = gamma.unsqueeze(1).expand_as(x_agg)
        return gamma

    def compute_gamma_squash(self, x_agg, edge_index_2hop, p=2):
        row2, col2 = edge_index_2hop
        diffs2 = (x_agg[row2] - x_agg[col2]).abs().pow(p).sum(dim=-1)
        num_nodes = x_agg.size(0)
        gamma2 = torch.zeros(num_nodes, device=x_agg.device)
        gamma2.index_add_(0, row2, diffs2)
        deg2 = torch.bincount(row2).float() + 1e-10
        gamma2 = gamma2 / deg2
        gamma2 = torch.tanh(gamma2)
        gamma2 = gamma2.unsqueeze(1).expand_as(x_agg)
        return gamma2

    def compute_abc(self, x, x_agg, x0, Gamma_smooth, Gamma_squash):
        denom = 1.0 + Gamma_smooth + Gamma_squash
        A = 1.0 / denom
        B = Gamma_smooth / denom
        C = Gamma_squash / denom
        return A, B, C


########################################################
# build_model & dataset
########################################################
def build_model(args):
    assert args.model in [
        'gin','gcn','gat','sage',
        'g2-gcn','g2-gat',
        'dual-gcn','dual-gat',
        'dual-hop-gcn','dual-hop-gat'
    ], f"Unknown model {args.model}"

    assert args.input_dim is not None, "Invalid input_dim"
    assert args.hidden_dim is not None, "Invalid hidden_dim"
    assert args.output_dim is not None, "Invalid output_dim"
    assert args.mpnn_layers is not None, "Invalid mpnn_layers"
    assert args.norm is not None, "Invalid normalisation"

    models_map = {
        'gin': GIN,
        'gcn': GCN,
        'gat': GAT,
        'sage': GraphSAGE
    }

    gating_map = {
        'g2-gcn': G2GCNModel,
        'g2-gat': G2GATModel,
        'dual-gcn': DualGate_GCNModel,
        'dual-gat': DualGate_GATModel,
        # 2-hop oversquash:
        'dual-hop-gcn': DualHopGCNModel,
        'dual-hop-gat': DualHopGATModel
    }

    if args.model in models_map:
        ModelClass = models_map[args.model]
        return ModelClass(
            in_channels=args.input_dim,
            hidden_channels=args.hidden_dim,
            out_channels=args.output_dim,
            num_layers=args.mpnn_layers,
            norm=args.norm
        )
    else:
        GatingClass = gating_map[args.model]
        return GatingClass(
            in_channels=args.input_dim,
            hidden_channels=args.hidden_dim,
            out_channels=args.output_dim,
            num_layers=args.mpnn_layers
        )


def build_dataset(args):
    assert args.dataset in ['TREE','RING','LOLLIPOP'], f"Unknown dataset {args.dataset}"

    dataset_factory = {
        'TREE': generate_tree_transfer_graph_dataset,
        'RING': generate_ring_transfer_graph_dataset,
        'LOLLIPOP': generate_lollipop_transfer_graph_dataset
    }
    dataset_configs = {
        'depth': args.synthetic_size,
        'nodes': args.synthetic_size,
        'classes': args.num_class,
        'samples': args.synth_train_size + args.synth_test_size,
        'arity': args.arity,
        'add_crosses': int(args.add_crosses)
    }
    return dataset_factory[args.dataset](**dataset_configs)
