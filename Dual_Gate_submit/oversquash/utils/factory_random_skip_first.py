#######################################
# factory_random.py
#######################################
import torch
import torch.nn as nn
import torch.nn.functional as F

# Original PyG MPNN classes
from torch_geometric.nn import GIN, GCN, GAT, GraphSAGE

# Data generation for synthetic tasks (if needed)
from data.ring_transfer import (
    generate_tree_transfer_graph_dataset,
    generate_ring_transfer_graph_dataset,
    generate_lollipop_transfer_graph_dataset
)

# For G2 and Dual gating classes:
from torch_geometric.nn import GCNConv, GATConv


###############################################################################
# 1) G2 Single-Gate Classes (GCN / GAT) with input projections
###############################################################################
class G2GCNModel(nn.Module):
    """
    Multi-layer GCN with G2 gating logic.
    Exactly as before, no changes here.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, norm=None):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels

        # 1) Input projection
        self.input_fc = nn.Linear(in_channels, hidden_channels, bias=False)

        # 2) GCN aggregator layers
        self.convs = nn.ModuleList([
            GCNConv(hidden_channels, hidden_channels, normalize=norm)
            for _ in range(num_layers)
        ])

        # 3) Final linear
        self.fc = nn.Linear(hidden_channels, out_channels, bias=True)

    def forward(self, x, edge_index):
        x = self.input_fc(x)  # shape [N, hidden_channels]

        for conv in self.convs:
            x_new = F.relu(conv(x, edge_index))
            tau = self.compute_g2_tau(x, x_new, edge_index)
            x = (1 - tau)*x + tau*x_new

        out = self.fc(x)
        return out

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
    """
    Single-gate G2 but aggregator is GATConv.
    Also unchanged.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, norm=None):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels

        self.input_fc = nn.Linear(in_channels, hidden_channels, bias=False)
        self.convs = nn.ModuleList([
            GATConv(hidden_channels, hidden_channels, heads=1)
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


###############################################################################
# 2) Dual-Gate Classes (GCN / GAT) with skip transform & input projections
#    with gating disabled on the first layer
###############################################################################
class DualGate_GCNModel(nn.Module):
    """
    Dual gating with aggregator = GCNConv, but on the first layer,
    gating is DISABLED => acts like plain GCN aggregator => can start near 1.
    Then from layer 1 onward, we re-enable gating logic.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, norm=None):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels

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

        for layer_idx, conv in enumerate(self.convs):
            x_agg = F.relu(conv(x, edge_index))

            if layer_idx == 0:
                # no gating => plain aggregator
                x = x_agg
            else:
                # normal dual gating
                Gamma_smooth = self.compute_gamma_smooth(x_agg, edge_index)
                # example random oversquash gamma
                Gamma_squash = 0.5 + 0.4 * torch.rand(x.size(0), device=x.device)
                Gamma_squash = Gamma_squash.unsqueeze(1).expand_as(x)

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

    def compute_abc(self, x, x_agg, x0, Gamma_smooth, Gamma_squash):
        denom = 1.0 + Gamma_smooth + Gamma_squash
        A = 1.0 / denom
        B = Gamma_smooth / denom
        C = Gamma_squash / denom
        return A, B, C


class DualGate_GATModel(nn.Module):
    """
    Dual gating with aggregator = GATConv, 
    also disabling gating on the first layer => plain aggregator pass layer=0.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels

        self.input_fc = nn.Linear(in_channels, hidden_channels, bias=False)
        self.skip_fc  = nn.Linear(hidden_channels, hidden_channels, bias=False)

        self.convs = nn.ModuleList([
            GATConv(hidden_channels, hidden_channels, heads=1)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(hidden_channels, out_channels, bias=True)

    def forward(self, x, edge_index, x0):
        x  = self.input_fc(x)
        x0 = self.input_fc(x0)

        for layer_idx, conv in enumerate(self.convs):
            x_agg = F.elu(conv(x, edge_index))

            if layer_idx == 0:
                # no gating => normal aggregator
                x = x_agg
            else:
                Gamma_smooth = self.compute_gamma_smooth(x_agg, edge_index)
                Gamma_squash = 0.5 + 0.4 * torch.rand(x.size(0), device=x.device)
                Gamma_squash = Gamma_squash.unsqueeze(1).expand_as(x)

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

    def compute_abc(self, x, x_agg, x0, Gamma_smooth, Gamma_squash):
        denom = 1.0 + Gamma_smooth + Gamma_squash
        A = 1.0 / denom
        B = Gamma_smooth / denom
        C = Gamma_squash / denom
        return A, B, C


#######################################
# 3) build_model & build_dataset
#######################################
def build_model(args):
    """
    The rest is the same as your code, 
    but now the Dual classes skip gating on layer=0 => can start near 1.
    """
    assert args.model in [
        'gin','gcn','gat','sage',
        'g2-gcn','g2-gat',
        'dual-gcn','dual-gat'
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
        'dual-gat': DualGate_GATModel
    }

    if args.model in models_map:
        # standard PyG model
        ModelClass = models_map[args.model]
        return ModelClass(
            in_channels=args.input_dim,
            hidden_channels=args.hidden_dim,
            out_channels=args.output_dim,
            num_layers=args.mpnn_layers,
            norm=args.norm
        )
    else:
        # gating approach
        GatingClass = gating_map[args.model]
        return GatingClass(
            in_channels=args.input_dim,
            hidden_channels=args.hidden_dim,
            out_channels=args.output_dim,
            num_layers=args.mpnn_layers
        )


def build_dataset(args):
    """
    Synthetic dataset if needed. 
    If you have real data, skip or remove this.
    """
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
