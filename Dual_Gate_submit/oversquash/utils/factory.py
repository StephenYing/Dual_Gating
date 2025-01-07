#######################################
# factory.py
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
    We do an 'input_fc' to map x from [N, in_channels] -> [N, hidden_channels].
    Then aggregator is GCNConv(hidden_channels, hidden_channels).
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, norm=None):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels

        # 1) Input projection: from in_channels -> hidden_channels
        self.input_fc = nn.Linear(in_channels, hidden_channels, bias=False)

        # 2) Define GCN layers: each from hidden_channels -> hidden_channels
        self.convs = nn.ModuleList([
            GCNConv(hidden_channels, hidden_channels, normalize=norm)
            for _ in range(num_layers)
        ])

        # 3) Final linear
        self.fc = nn.Linear(hidden_channels, out_channels, bias=True)

    def forward(self, x, edge_index):
        # 1) Project x from [N, in_channels] to [N, hidden_channels]
        x = self.input_fc(x)  # shape: [N, hidden_channels]

        for conv in self.convs:
            x_new = F.relu(conv(x, edge_index))
            tau = self.compute_g2_tau(x, x_new, edge_index)
            x = (1 - tau) * x + tau * x_new

        out = self.fc(x)  # [N, out_channels]
        return out

    def compute_g2_tau(self, old_x, new_x, edge_index, p=2):
        """
        G2 gating: tau_i = tanh(mean_{j in N(i)} || new_x_i - new_x_j ||^p).
        """
        row, col = edge_index
        diffs = (new_x[row] - new_x[col]).abs().pow(p).sum(dim=-1)

        num_nodes = old_x.size(0)
        tau_vals = torch.zeros(num_nodes, device=old_x.device)
        tau_vals.index_add_(0, row, diffs)

        deg = torch.bincount(row).float() + 1e-10
        tau_vals = tau_vals / deg
        tau_vals = torch.tanh(tau_vals).unsqueeze(-1)  # shape [N,1]
        return tau_vals


class G2GATModel(nn.Module):
    """
    Single-gate G2 but aggregator is GATConv.
    We also use an input_fc to unify dims: [N, in_channels]->[N, hidden_channels].
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, norm=None):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels

        # 1) Input projection
        self.input_fc = nn.Linear(in_channels, hidden_channels, bias=False)

        # 2) GAT layers
        self.convs = nn.ModuleList([
            GATConv(hidden_channels, hidden_channels, heads=1)
            for _ in range(num_layers)
        ])

        self.fc = nn.Linear(hidden_channels, out_channels, bias=True)

    def forward(self, x, edge_index):
        x = self.input_fc(x)  # [N, hidden_channels]

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


###############################################################################
# 2) Dual-Gate Classes (GCN / GAT) with skip transform & input projections
###############################################################################
class DualGate_GCNModel(nn.Module):
    """
    Dual gating with aggregator = GCNConv.
    forward(x, edge_index, x0), but we also do an input_fc so x, x0 all become [N, hidden_channels].
    Here, we compute both Gamma_smooth and Gamma_squash from local differences,
    removing the random oversquash approach.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, norm=None):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels

        # 1) We project from in_channels->hidden_channels for aggregator,
        #    plus a skip_fc from hidden_channels->hidden_channels if needed.
        self.input_fc = nn.Linear(in_channels, hidden_channels, bias=False)
        self.skip_fc  = nn.Linear(hidden_channels, hidden_channels, bias=False)

        # 2) GCN layers (all from hidden_channels->hidden_channels)
        self.convs = nn.ModuleList([
            GCNConv(hidden_channels, hidden_channels, normalize=norm)
            for _ in range(num_layers)
        ])

        self.fc = nn.Linear(hidden_channels, out_channels, bias=True)

    def forward(self, x, edge_index, x0):
        """
        x: shape [N, in_channels]
        x0: shape [N, in_channels] for skip
        """
        # Project both x & x0 to hidden_dims
        x  = self.input_fc(x)    # [N, hidden_channels]
        x0 = self.input_fc(x0)   # also [N, hidden_channels]

        for conv in self.convs:
            x_agg = F.relu(conv(x, edge_index))

            # 1) Gamma_smooth from local differences
            Gamma_smooth = self.compute_gamma_local_diff(x_agg, edge_index)

            # 2) Gamma_squash also from local differences in x_agg, 
            #    so oversquash gating is data-driven (not random).
            Gamma_squash = self.compute_gamma_local_diff(x_agg, edge_index)

            A, B, C = self.compute_abc(x, x_agg, x0, Gamma_smooth, Gamma_squash)
            x_skip = self.skip_fc(x0)

            x = A*x + B*x_agg + C*x_skip

        out = self.fc(x)
        return out

    def compute_gamma_local_diff(self, x_agg, edge_index, p=2):
        """
        Node-wise gating from neighbor differences in x_agg (just like G2).
        """
        row, col = edge_index
        diffs = (x_agg[row] - x_agg[col]).abs().pow(p).sum(dim=-1)

        num_nodes = x_agg.size(0)
        gamma = torch.zeros(num_nodes, device=x_agg.device)
        gamma.index_add_(0, row, diffs)

        deg = torch.bincount(row).float() + 1e-10
        gamma = gamma / deg
        gamma = torch.tanh(gamma)
        gamma = gamma.unsqueeze(1).expand_as(x_agg)  # shape [N,d]
        return gamma

    def compute_abc(self, x, x_agg, x0, Gamma_smooth, Gamma_squash):
        denom = 1.0 + Gamma_smooth + Gamma_squash
        A = 1.0 / denom
        B = Gamma_smooth / denom
        C = Gamma_squash / denom
        return A, B, C


class DualGate_GATModel(nn.Module):
    """
    aggregator = GATConv, with forward(x, edge_index, x0).
    We'll do an input_fc for both x & x0, then aggregator sees shape [N, hidden_channels].
    Similarly, both Gamma_smooth and Gamma_squash are from neighbor diffs.
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
        # Project x, x0 to hidden_dims
        x  = self.input_fc(x)
        x0 = self.input_fc(x0)

        for conv in self.convs:
            x_agg = F.relu(conv(x, edge_index))

            # 1) Gamma_smooth from neighbor diffs
            Gamma_smooth = self.compute_gamma_local_diff(x_agg, edge_index)

            # 2) oversquash gating also from local diffs
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


#######################################
# 3) build_model & build_dataset
#######################################
def build_model(args):
    """
    Builds either a standard GIN/GCN/GAT/SAGE from PyG, or
    a gating model (G2 / Dual) from above, depending on args.model.

    Expect:
      args.model in [
         'gin','gcn','gat','sage',
         'g2-gcn','g2-gat',
         'dual-gcn','dual-gat'
      ]
      args.input_dim (e.g., 37),
      args.hidden_dim (e.g., 5),
      args.output_dim,
      args.mpnn_layers,
      args.norm (string)
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
    Builds a synthetic dataset using ring, tree, or lollipop graphs
    if desired. Adjust or remove if you have your own real dataset.
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
