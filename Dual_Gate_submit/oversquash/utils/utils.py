#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: On Oversquashing project authors 
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from functorch import vmap
from torch.utils.data.sampler import Sampler
import random
import pandas as pd
import networkx as nx

eps_min = 1e-12
eps_max = 1e+12


def sparse_to_tuple(sparse_mx):
    """Convert a scipy sparse matrix to a (coords, values, shape) tuple."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def coo2tensor(A):
    """Converts a scipy COOrdinate matrix A into a PyTorch sparse tensor."""
    assert(sp.isspmatrix_coo(A))
    idxs = torch.LongTensor(np.vstack((A.row, A.col)))
    vals = torch.FloatTensor(A.data)
    return torch.sparse_coo_tensor(idxs, vals, size=A.shape, requires_grad=False)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def degree_normalize_adjacency(A):
    """
    Returns D^-1 * A * D^-1 for an adjacency matrix A.
    A must be square (shape[0] == shape[1]).
    """
    assert A.shape[0] == A.shape[1]
    d = np.sum(A, axis=1)
    d = 1/np.sqrt(d)
    D = np.diag(d)
    return sp.coo_matrix(D @ A @ D)


def preprocess_adj(adj):
    """Preprocessing of adjacency for a simple GCN model, plus conversion to tuple."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, placeholders):
    """Construct feed dictionary for GCN-Align (placeholder-based)."""
    feed_dict = {}
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({
        placeholders['support'][i]: support[i]
        for i in range(len(support))
    })
    return feed_dict


def chebyshev_polynomials(adj, k):
    """
    Calculate Chebyshev polynomials up to order k.
    Return a list of sparse matrices (tuple representation).
    """
    print(f"Calculating Chebyshev polynomials up to order {k}...")

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = []
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k + 1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


def loadfile(fn, num=1):
    """Load a file and return a list of tuple containing 'num' integers in each line."""
    print('loading a file...', fn)
    ret = []
    with open(fn, encoding='utf-8') as f:
        for line in f:
            th = line.strip().split('\t')
            x = []
            for i in range(num):
                x.append(int(th[i]))
            ret.append(tuple(x))
    return ret


def get_ent2id(fns):
    """Creates a dictionary ent->id from multiple files."""
    ent2id = {}
    for fn in fns:
        with open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                th = line.strip().split('\t')
                ent2id[th[1]] = int(th[0])
    return ent2id


def loadattr(fns, e, ent2id):
    """
    Loads attributes from file list 'fns'.
    Returns a sparse matrix of shape (e, num_features).
    """
    cnt = {}
    for fn in fns:
        with open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                th = line.strip().split('\t')
                if th[0] not in ent2id:
                    continue
                for i in range(1, len(th)):
                    cnt[th[i]] = cnt.get(th[i], 0) + 1

    fre = [(k, cnt[k]) for k in sorted(cnt, key=cnt.get, reverse=True)]
    num_features = min(len(fre), 2000)
    attr2id = {}
    for i in range(num_features):
        attr2id[fre[i][0]] = i

    M = {}
    for fn in fns:
        with open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                th = line.strip().split('\t')
                if th[0] in ent2id:
                    for i in range(1, len(th)):
                        if th[i] in attr2id:
                            M[(ent2id[th[0]], attr2id[th[i]])] = 1.0

    row, col, data = [], [], []
    for key, val in M.items():
        row.append(key[0])
        col.append(key[1])
        data.append(val)
    return sp.coo_matrix((data, (row, col)), shape=(e, num_features))


def get_dic_list(e, KG):
    M = {}
    for tri in KG:
        if tri[0] == tri[2]:
            continue
        M[(tri[0], tri[2])] = 1
        M[(tri[2], tri[0])] = 1
    dic_list = {i: [] for i in range(e)}
    for pair in M:
        dic_list[pair[0]].append(pair[1])
    return dic_list


def func(KG):
    head = {}
    cnt = {}
    for tri in KG:
        r = tri[1]
        head.setdefault(r, set()).add(tri[0])
        cnt[r] = cnt.get(r, 0) + 1
    r2f = {}
    for r in cnt:
        r2f[r] = len(head[r]) / cnt[r]
    return r2f


def ifunc(KG):
    tail = {}
    cnt = {}
    for tri in KG:
        r = tri[1]
        tail.setdefault(r, set()).add(tri[2])
        cnt[r] = cnt.get(r, 0) + 1
    r2if = {}
    for r in cnt:
        r2if[r] = len(tail[r]) / cnt[r]
    return r2if


def get_weighted_adj(e, KG):
    """
    Build weighted adjacency from KG edges, using some min factors (0.3).
    """
    r2f = func(KG)
    r2if = ifunc(KG)
    M = {}
    for tri in KG:
        if tri[0] == tri[2]:
            continue
        if (tri[0], tri[2]) not in M:
            M[(tri[0], tri[2])] = max(r2if[tri[1]], 0.3)
        else:
            M[(tri[0], tri[2])] += max(r2if[tri[1]], 0.3)

        if (tri[2], tri[0]) not in M:
            M[(tri[2], tri[0])] = max(r2f[tri[1]], 0.3)
        else:
            M[(tri[2], tri[0])] += max(r2f[tri[1]], 0.3)

    row, col, data = [], [], []
    for (r, c), val in M.items():
        row.append(c)
        col.append(r)
        data.append(val)
    return sp.coo_matrix((data, (row, col)), shape=(e, e))


def get_ae_input(attr):
    return sparse_to_tuple(sp.coo_matrix(attr))


def set_seed(seed):
    """Set the random seed for torch, numpy, and random."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def reset_wandb_env():
    """Clear out any WANDB_ environment variables except for a few protected ones."""
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]


def namespace_to_command(namespace: str, exp_file: str) -> str:
    """Convert a namespace string to a command line for 'exp_file'."""
    namespace = namespace.replace("Namespace(", "").replace(")", "")
    args_split = namespace.split(", ")

    args = ""
    filter_out = ["device", "input_dim", "output_dim", "edge_dim", "sha", "index"]
    for arg in args_split:
        if any(substring in arg for substring in filter_out):
            continue
        name, value = arg.split("=")
        arg_formatted = f"--{name} {value}"
        args += f"{arg_formatted} "
    return f"python exp/{exp_file} {args}"


class CustomSampler(Sampler[int]):
    """
    Samples elements randomly from a given list of indices, without replacement.
    If train=True, we shuffle, else keep the order.
    """
    def __init__(self, indices: list, train: bool) -> None:
        self.indices = indices
        self.train = train
        
    def __iter__(self):
        if self.train:
            perm = torch.randperm(len(self.indices))
            for i in perm:
                yield self.indices[i]
        else:
            for i in self.indices:
                yield i

    def __len__(self) -> int:
        return len(self.indices)


def smooth_plot(x, y=None, ax=None, label='', halflife=10):
    """
    Plots a smoothed line (exponential moving average) for (x,y) data.
      - x: x-axis data (or if y=None, x is the data to plot vs index)
      - y: optional y-axis data
      - ax: matplotlib axes
      - label: legend label
      - halflife: smoothing factor for pd.Series(...).ewm(halflife=..)
    """
    if ax is None:
        ax = plt.gca()

    # Distinguish between x-only vs (x,y)
    if y is None:
        y_int = x
        x_vals = np.arange(len(y_int))
    else:
        y_int = y
        x_vals = np.asarray(x)

    # We'll let Matplotlib pick the color automatically
    color = None

    # 1) Exponential moving average
    series = pd.Series(y_int)
    y_ewm = series.ewm(halflife=halflife)
    y_mean = y_ewm.mean().values
    y_std  = y_ewm.std().values

    # 2) Plot main line
    ax.plot(x_vals, y_mean, label=label, color=color)
    # 3) Fill for +/- std
    ax.fill_between(x_vals, y_mean - y_std, y_mean + y_std,
                    color=color, alpha=0.15)


def get_resistances(graph, reference_node):
    """
    Returns a dict {node: resistance_distance(reference_node, node)} for each node in 'graph'.
    Requires that 'graph' is a NetworkX graph with a single connected component 
    or a function 'nx.resistance_distance' that handles partial connectivity gracefully.
    """
    resistances = {}
    for node in graph.nodes:
        if node != reference_node:
            resistances[node] = nx.resistance_distance(graph, reference_node, node)
        else:
            resistances[node] = 0
    return resistances
