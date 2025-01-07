#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A single file combining:
  1) ring_transfer.py (graph generators)
  2) utils.py (helper utilities)
  3) factory.py (g2/dual gating classes + build_model)
  4) The main 'signal_propagation.py' script

You can run:
  python signal_propagation_all_in_one.py
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

# For PyTorch Geometric
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
from torch_geometric.nn import GIN, GCN, GAT, GraphSAGE, GCNConv, GATConv

from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
from typing import List
# If you need 'functorch', ensure it's installed.
# from functorch import vmap


###############################################################################
# (A) ring_transfer.py content
###############################################################################

def generate_ring_lookup_graph(nodes:int):
    """
    (Deprecated) Generate a dictionary lookup ring graph.
    """
    if nodes <= 1: 
        raise ValueError("Minimum of two nodes required")

    keys = np.arange(1, nodes)
    vals = np.random.permutation(nodes - 1)

    oh_keys = np.array(LabelBinarizer().fit_transform(keys))
    oh_vals = np.array(LabelBinarizer().fit_transform(vals))

    oh_all = np.concatenate((oh_keys, oh_vals), axis=-1)
    x = np.empty((nodes, oh_all.shape[1]))
    x[1:, :] = oh_all

    key_idx = random.randint(0, nodes - 2)
    val = vals[key_idx]

    x[0, :] = 0
    x[0, :oh_keys.shape[1]] = oh_keys[key_idx]

    x = torch.tensor(x, dtype=torch.float32)

    edge_index = []
    for i in range(nodes - 1):
        edge_index.append([i, i + 1])
        edge_index.append([i + 1, i])
    edge_index.append([0, nodes - 1])
    edge_index.append([nodes - 1, 0])

    edge_index = np.array(edge_index, dtype=np.compat.long).T
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    mask = torch.zeros(nodes, dtype=torch.bool)
    mask[0] = 1
    y = torch.tensor([val], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, mask=mask, y=y)


def generate_ringlookup_graph_dataset(nodes:int, samples:int=10000):
    """
    Build a dataset of ring-lookup graphs.
    """
    if nodes <= 1:
        raise ValueError("Minimum of two nodes required")
    dataset = []
    for _ in range(samples):
        graph = generate_ring_lookup_graph(nodes)
        dataset.append(graph)
    return dataset


def generate_ring_transfer_graph(nodes, target_label, add_crosses: bool):
    if nodes <= 1: 
        raise ValueError("Minimum of two nodes required")
    opposite_node = nodes // 2

    x = np.ones((nodes, len(target_label)))
    x[0, :] = 0.0
    x[opposite_node, :] = target_label
    x = torch.tensor(x, dtype=torch.float32)

    edge_index = []
    for i in range(nodes-1):
        edge_index.append([i, i + 1])
        edge_index.append([i + 1, i])
        if add_crosses and i < opposite_node:
            edge_index.append([i, nodes - 1 - i])
            edge_index.append([nodes - 1 - i, i])
            if nodes + 1 - i < nodes:
                edge_index.append([i, nodes + 1 - i])
                edge_index.append([nodes + 1 - i, i])

    edge_index.append([0, nodes - 1])
    edge_index.append([nodes - 1, 0])

    edge_index = np.array(edge_index, dtype=np.compat.long).T
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    mask = torch.zeros(nodes, dtype=torch.bool)
    mask[0] = 1
    y = torch.tensor([np.argmax(target_label)], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, mask=mask, y=y)


def generate_ring_transfer_graph_dataset(nodes:int, add_crosses:bool=False, classes:int=5, samples:int=10000, **kwargs):
    if nodes <= 1: 
        raise ValueError("Minimum of two nodes required")
    dataset = []
    samples_per_class = samples // classes
    for i in range(samples):
        label = i // samples_per_class
        target_class = np.zeros(classes)
        target_class[label] = 1.0
        graph = generate_ring_transfer_graph(nodes, target_class, add_crosses)
        dataset.append(graph)
    return dataset


def generate_tree_transfer_graph(depth:int, target_label:List[int], arity:int):
    if depth <= 0:
        raise ValueError("Minimum of depth one")
    num_nodes = int((arity ** (depth + 1) - 1) / (arity - 1))
    target_node = num_nodes - 1

    x = np.ones((num_nodes, len(target_label)))
    x[0, :] = 0.0
    x[target_node, :] = target_label
    x = torch.tensor(x, dtype=torch.float32)

    edge_index = []
    last_child_counter = 0

    for i in range(num_nodes - arity ** depth + 1):
        for child in range(1, arity + 1):
            if last_child_counter + child > num_nodes - 1:
                break
            edge_index.append([i, last_child_counter + child])
            edge_index.append([last_child_counter + child, i])
        last_child_counter += arity

    edge_index = np.array(edge_index, dtype=np.compat.long).T
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[0] = 1
    y = torch.tensor([np.argmax(target_label)], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, mask=mask, y=y)


def generate_tree_transfer_graph_dataset(depth:int, arity:int, classes:int=5, samples:int=10000, **kwargs):
    dataset = []
    samples_per_class = samples // classes
    for i in range(samples):
        label = i // samples_per_class
        target_class = np.zeros(classes)
        target_class[label] = 1.0
        graph = generate_tree_transfer_graph(depth, target_class, arity)
        dataset.append(graph)
    return dataset


def generate_lollipop_transfer_graph(nodes:int, target_label:List[int]):
    if nodes <= 1: 
        raise ValueError("Minimum of two nodes required")    
    x = np.ones((nodes, len(target_label)))
    x[0, :] = 0.0
    x[nodes - 1, :] = target_label
    x = torch.tensor(x, dtype=torch.float32)

    edge_index = []

    for i in range(nodes // 2):
        for j in range(nodes // 2):
            if i == j:
                continue
            edge_index.append([i, j])
            edge_index.append([j, i])

    for i in range(nodes // 2, nodes - 1):
        edge_index.append([i, i+1])
        edge_index.append([i+1, i])

    edge_index.append([nodes // 2 - 1, nodes // 2])
    edge_index.append([nodes // 2, nodes // 2 - 1])

    edge_index = np.array(edge_index, dtype=np.compat.long).T
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    mask = torch.zeros(nodes, dtype=torch.bool)
    mask[0] = 1
    y = torch.tensor([np.argmax(target_label)], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, mask=mask, y=y)


def generate_lollipop_transfer_graph_dataset(nodes:int, classes:int=5, samples:int=10000, **kwargs):
    if nodes <= 1: 
        raise ValueError("Minimum of two nodes required")
    dataset = []
    samples_per_class = samples // classes
    for i in range(samples):
        label = i // samples_per_class
        target_class = np.zeros(classes)
        target_class[label] = 1.0
        graph = generate_lollipop_transfer_graph(nodes, target_class)
        dataset.append(graph)
    return dataset


###############################################################################
# (B) utils.py content
###############################################################################

eps_min = 1e-12
eps_max = 1e+12

def smooth_plot(x, y=None, ax=None, label='', halflife=10):
    """
    Plot a smoothed curve of (x, y) or just x.
    """
    import pandas as pd
    if y is None:
        y_int = x
    else:
        y_int = y

    x_ewm = pd.Series(y_int).ewm(halflife=halflife)
    color = next(plt.gca()._get_lines.prop_cycler)['color']
    if y is None:
        ax.plot(x_ewm.mean(), label=label, color=color)
        ax.fill_between(np.arange(x_ewm.mean().shape[0]),
                        x_ewm.mean() + x_ewm.std(),
                        x_ewm.mean() - x_ewm.std(),
                        color=color, alpha=0.15)
    else:
        ax.plot(x, x_ewm.mean(), label=label, color=color)
        ax.fill_between(x,
                        y_int + x_ewm.std(),
                        y_int - x_ewm.std(),
                        color=color, alpha=0.15)

def get_resistances(graph, reference_node):
    """
    Compute the 'resistance distance' from reference_node to each other node,
    using networkx's built-in resistance_distance.
    May require a connected or strongly connected graph.
    """
    resistances = {}
    for node in graph.nodes:
        if node != reference_node:
            # if graph is not connected, networkx.resistance_distance can fail
            resistances[node] = nx.resistance_distance(graph, reference_node, node)
        else:
            resistances[node] = 0
    return resistances


###############################################################################
# (C) factory.py content: G2, Dual gating classes, plus build_model
###############################################################################

class G2GCNModel(nn.Module):
    """
    Already defined above, repeated for clarity if needed...
    (We keep them here to show the final single-file structure.)
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, norm=None):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels

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
        tau_vals /= deg
        tau_vals = torch.tanh(tau_vals).unsqueeze(-1)
        return tau_vals


class G2GATModel(nn.Module):
    """
    Single-gate G2 with GAT aggregator (all in hidden_channels).
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
        tau_vals /= deg
        tau_vals = torch.tanh(tau_vals).unsqueeze(-1)
        return tau_vals


class DualGate_GCNModel(nn.Module):
    """
    Dual gating (A,B,C) with aggregator=GCN.
    forward(x, edge_index, x0).
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

        for conv in self.convs:
            x_agg = F.relu(conv(x, edge_index))
            Gamma_smooth = self.compute_gamma_smooth(x_agg, edge_index)
            Gamma_squash = self.compute_gamma_squash_global(x.size(0), device=x.device)
            Gamma_squash = Gamma_squash.unsqueeze(1).expand_as(x)
            A, B, C = self.compute_abc(x, x_agg, x0, Gamma_smooth, Gamma_squash)
            x_skip = self.skip_fc(x0)
            x = A*x + B*x_agg + C*x_skip

        return self.fc(x)

    def compute_gamma_smooth(self, x_agg, edge_index, p=2):
        row, col = edge_index
        diffs = (x_agg[row] - x_agg[col]).abs().pow(p).sum(dim=-1)
        gamma = torch.zeros(x_agg.size(0), device=x_agg.device)
        gamma.index_add_(0, row, diffs)
        deg = torch.bincount(row).float() + 1e-10
        gamma /= deg
        gamma = torch.tanh(gamma).unsqueeze(1).expand_as(x_agg)
        return gamma

    def compute_gamma_squash_global(self, x, p=2.):
        # gamma_squash(i)=1 - tanh(||x_i - mean||^p)
        global_mean = x.mean(dim=0, keepdim=True)
        d = (x - global_mean).abs().pow(p).sum(dim=-1)
        # shape [N]
        d_tanh=torch.tanh(d)
        gamma_squash=1. - d_tanh
        return gamma_squash.unsqueeze(-1)
    
    def compute_abc(self, x, x_agg, x0, Gamma_smooth, Gamma_squash):
        denom = 1.0 + Gamma_smooth + Gamma_squash
        A = 1.0 / denom
        B = Gamma_smooth / denom
        C = Gamma_squash / denom
        return A, B, C


class DualGate_GATModel(nn.Module):
    """
    aggregator=GAT, dual gating => forward(x, edge_index, x0)
    with an input_fc, skip_fc, etc.
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

        for conv in self.convs:
            x_agg = F.elu(conv(x, edge_index))
            Gamma_smooth = self.compute_gamma_smooth(x_agg, edge_index)
            Gamma_squash = self.compute_gamma_squash_global(x.size(0), device=x.device)
            A, B, C = self.compute_abc(x, x_agg, x0, Gamma_smooth, Gamma_squash)
            x_skip = self.skip_fc(x0)
            x = A*x + B*x_agg + C*x_skip
        return self.fc(x)

    def compute_gamma_smooth(self, x_agg, edge_index, p=2):
        row, col = edge_index
        diffs = (x_agg[row] - x_agg[col]).abs().pow(p).sum(dim=-1)
        gamma = torch.zeros(x_agg.size(0), device=x_agg.device)
        gamma.index_add_(0, row, diffs)
        deg = torch.bincount(row).float() + 1e-10
        gamma /= deg
        gamma = torch.tanh(gamma).unsqueeze(1).expand_as(x_agg)
        return gamma

    def compute_gamma_squash_global(self, x, p=2.):
        # gamma_squash(i)=1 - tanh(||x_i - mean||^p)
        global_mean = x.mean(dim=0, keepdim=True)
        d = (x - global_mean).abs().pow(p).sum(dim=-1)
        # shape [N]
        d_tanh=torch.tanh(d)
        gamma_squash=1. - d_tanh
        return gamma_squash.unsqueeze(-1)

    def compute_abc(self, x, x_agg, x0, Gamma_smooth, Gamma_squash):
        denom = 1.0 + Gamma_smooth + Gamma_squash
        A = 1.0 / denom
        B = Gamma_smooth / denom
        C = Gamma_squash / denom
        return A, B, C


def build_model(args):
    """
    Builds either a standard GIN/GCN/GAT/SAGE from PyG, or
    a gating model (G2 / Dual).
    """
    assert args.model in [
        'gin','gcn','gat','sage','g2-gcn','g2-gat','dual-gcn','dual-gat'
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
    """
    Optionally build synthetic ring/tree/lollipop datasets, if desired.
    Adjust or remove if you have your own real dataset.
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


###############################################################################
# (D) The main signal_propagation script
###############################################################################
def initialize_architecture(arch, dataset_item, layers=10, dim_h=5):
    """
    Create a simple 'args' object for build_model(...).
    We'll guess input_dim from dataset_item.x, then build the model.
    """
    from types import SimpleNamespace
    args = SimpleNamespace()

    args.model = arch
    args.input_dim = dataset_item.x.size(1)
    args.output_dim = dataset_item.x.size(1)
    args.hidden_dim = dim_h
    args.mpnn_layers = layers
    args.norm = "batch_norm"
    # optional: you can remove the above assertion if you truly want 'norm=None'

    model = build_model(args)
    return model


def process_graph_data(dataset_item, arch, num_vertices_sampled=10):
    model = initialize_architecture(arch, dataset_item)
    model.eval()

    G = to_networkx(dataset_item, to_undirected=True)

    # If the graph is disconnected, skip
    if not nx.is_connected(G):
        print("Skipping a disconnected graph.")
        return None  

    # Floyd-Warshall for all pairs
    distances = nx.floyd_warshall_numpy(G)

    pairs_runaway = []
    for _ in range(num_vertices_sampled):
        source = np.random.randint(0, len(G))
        max_t_A_st = distances[source].max()

        x = torch.zeros_like(dataset_item.x)
        x[source] = torch.randn_like(dataset_item.x[source])
        x[source] = x[source].softmax(dim=-1)

        edge_index = dataset_item.edge_index
        # Dual gating => forward(x, edge_index, x0)
        if arch in ['dual-gcn', 'dual-gat']:
            out = model(x, edge_index, x)
        else:
            out = model(x, edge_index)

        out = (out / out.sum()).cpu().detach().numpy()
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

        acc = 0.0
        for j in range(len(out)):
            acc += out[j] * distances[j, source]
        propagation = ((1/max_t_A_st) * acc).mean()

        # Effective resistances
        # If the graph is not fully connected or if reference_node is unreachable, 
        # networkx.resistance_distance can fail.
        total_effective_resistance = sum(get_resistances(G, source).values())
        pairs_runaway.append((total_effective_resistance, propagation))

    return pairs_runaway


def main():
    # If you'd rather use a real dataset, pick TUDataset name:
    # e.g. name='PROTEINS', 'NCI1', 'MUTAG', etc.
    DATASET_NAME = 'PROTEINS'
    dataset = TUDataset(root='.', name=DATASET_NAME)

    ARCHS = ['gcn','gat','g2-gcn','g2-gat','dual-gcn','dual-gat']
    pairs = {arch: [] for arch in ARCHS}

    for i, data in enumerate(dataset):
        try:
            # for each architecture
            for arch in ARCHS:
                arch_runaway = process_graph_data(data, arch, num_vertices_sampled=10)
                if not arch_runaway: 
                    # e.g. disconnected => skip
                    continue
                mean_runaway = np.array(arch_runaway).mean(axis=0).tolist()  # shape (2,)
                pairs[arch].append(tuple(mean_runaway))
        except Exception as e:
            print(f"Error processing graph {i}: {str(e)}")

    # Plot
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    titles = ["GCN","GAT","G2-GCN","G2-GAT","Dual-GCN","Dual-GAT"]

    for ax, arch, title in zip(axs.flat, ARCHS, titles):
        data_array = np.array(pairs[arch])
        if len(data_array) == 0:
            ax.set_title(f"{title} (No Data)", fontsize=20)
            continue
        sorted_data = data_array[data_array[:,0].argsort()]
        x = sorted_data[:,0]
        y = sorted_data[:,1]
        from utils.utils import smooth_plot  # or define smooth_plot above
        smooth_plot(x, y, ax=ax, halflife=2)
        ax.set_title(title, fontsize=20)
        ax.set_ylim(0,1)
        ax.set_xlim(0,1)


    fig.text(0.5, 0.04, 'Normalized Total Effective Resistance', size=20, ha='center', va='center')
    fig.text(0.06, 0.5, 'Signal Propagation', size=20, ha='center', va='center', rotation='vertical')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
