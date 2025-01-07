import os 
import sys
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx

# Adjust this path as needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.utils import smooth_plot, get_resistances, set_seed
from utils.factory_random import build_model  # <- your updated factory with "forward(x, edge_index)" etc.

ROOT_DIR = '.'
DATASET_NAME = 'NCI1'
NUM_VERTICES_SAMPLED = 10

ARCHS = [
    'gcn',
    'gat',
    'g2-gcn',
    'g2-gat',
    'dual-gcn',
    'dual-gat'
]

def initialize_architecture(arch, dataset_item, layers=10, dim_h=5):
    """
    Create a simple 'args' object for build_model(...).
    We'll guess input_dim from dataset_item.x, then build the model.
    """
    from types import SimpleNamespace
    args = SimpleNamespace()

    # 1) Which aggregator/gating approach
    args.model = arch  # e.g., 'g2-gcn'

    # 2) Dimensions
    args.input_dim = dataset_item.x.size(1)   # e.g., node feature dim
    args.output_dim = dataset_item.x.size(1)  # just for demonstration
    args.hidden_dim = dim_h

    # 3) Number of layers
    args.mpnn_layers = layers

    # 4) Normalization (the updated code requires a non-None string)
    args.norm = "batch_norm"

    # Build & return
    model = build_model(args)
    return model


def process_graph_data(dataset_item, arch):
    """
    For a single PyG 'Data' item and a chosen arch,
    run multiple forward passes with different 'source' setups,
    measure 'signal propagation' vs. 'effective resistance'.
    """
    model = initialize_architecture(arch, dataset_item)
    model.eval()  # no training here

    # We'll use networkx for distance calculations
    G = to_networkx(dataset_item, to_undirected=True)
    # Check if G is connected
    if not nx.is_connected(G):
        # Possibly just skip this graph
        print("Skipping a disconnected graph.")
        return None

    # Otherwise we do distances
    distances = nx.floyd_warshall_numpy(G)

    pairs_runaway = []
    for _ in range(NUM_VERTICES_SAMPLED):
        # Pick a random source node
        source = np.random.randint(0, len(G))
        max_t_A_st = distances[source].max()

        # Prepare a new feature matrix (so only the 'source' node is nonzero)
        x = torch.zeros_like(dataset_item.x)
        x[source] = torch.randn_like(dataset_item.x[source])
        x[source] = x[source].softmax(dim=-1)

        # Now we call the model. Because the updated GNN classes expect
        # either (x, edge_index) or (x, edge_index, x0),
        # we must handle "dual" vs. "non-dual" differently:
        edge_index = dataset_item.edge_index

        if arch in ['dual-gcn', 'dual-gat']:
            # DualGate_X models => forward(self, x, edge_index, x0)
            # Let's use the same 'x' as x0 or the original data.x
            # If you want the "original input" as skip, you can do dataset_item.x or whatever.
            # For demonstration, let's pass x itself as x0 here:
            out = model(x, edge_index, x)  
        else:
            # GCN/GAT/G2-GCN/G2-GAT => forward(self, x, edge_index)
            out = model(x, edge_index)

        # Convert out to numpy, handle nans
        out = (out / out.sum()).cpu().detach().numpy()
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

        # "Propagation" measure => sum_{j}( out[j] * dist(j, source) )
        acc = 0.0
        for j in range(len(out)):
            acc += out[j] * distances[j, source]
        propagation = ((1/max_t_A_st) * acc).mean()

        # Effective resistance from Di Giovanni's snippet
        total_effective_resistance = sum(get_resistances(G, source).values())

        pairs_runaway.append((total_effective_resistance, propagation))

    return pairs_runaway


def plot_data(ax, data, title):
    x = MinMaxScaler().fit_transform(data[:, 0].reshape(-1, 1)).flatten()
    y = MinMaxScaler().fit_transform(data[:, 1].reshape(-1, 1)).flatten()
    smooth_plot(x=x, y=y, ax=ax, halflife=2)
    ax.set_title(title, fontsize=20)


def main():
    set_seed(0)
    dataset = TUDataset(root=ROOT_DIR, name=DATASET_NAME)
    print("Original dataset size:", len(dataset))

    # For demonstration, keep only the first 200 graphs:
    dataset = dataset[:200]
    print("Reduced dataset size:", len(dataset))
    pairs = {arch: [] for arch in ARCHS}

    from tqdm import tqdm
    for i, data in tqdm(enumerate(dataset), total=len(dataset), desc="Processing Graphs"):
        try:
            for arch in ARCHS:
                arch_runaway = process_graph_data(data, arch)
                if arch_runaway is None or len(arch_runaway) == 0:
                    # Disconnected or empty => skip
                    continue
                mean_runaway = np.mean(arch_runaway, axis=0)
                # If you expect mean_runaway to be 1D, do e.g. shape = (2,)
                # then pairs[arch].append(tuple(mean_runaway))
                pairs[arch].append(tuple(mean_runaway))
        except Exception as e:
            print(f"Error processing graph {i}: {str(e)}")

    # We have 6 methods, so let's do a 2x3 subplots
    fig, axs = plt.subplots(3, 2, figsize=(12, 10))
    titles = ["GCN","GAT","G2-GCN","G2-GAT","Dual-GCN","Dual-GAT"]

    for ax, arch, title in zip(axs.flat, ARCHS, titles):
        data_array = np.array(pairs[arch])
        if len(data_array) == 0:
            ax.set_title(f"{title} (No Data)", fontsize=20)
            ax.set_xlabel("Normalized Total Effective Resistance")
            ax.set_ylabel("Signal Propagation")
            ax.set_xlim(0,1)
            ax.set_ylim(0,1)
            continue

        sorted_data = data_array[data_array[:, 0].argsort()]
        plot_data(ax, sorted_data, title)
        ax.set_xlabel("Normalized Total Effective Resistance")
        ax.set_ylabel("Signal Propagation")
        ax.set_ylim(0,1)
        
    # fig.text(0.5, 0.04, 'Normalized Total Effective Resistance', size=20, ha='center', va='center')
    # fig.text(0.06, 0.5, 'Signal Propagation', size=20, ha='center', va='center', rotation='vertical')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
