# ── SECTION 0: Auto-install dependencies ────────────────────────
import subprocess, sys, shutil, os

# Install native METIS library
if shutil.which('apt-get') and not shutil.which('gpmetis'):
    subprocess.check_call(['sudo', 'apt-get', 'install', '-y', '-qq', 'libmetis-dev'])
os.environ.setdefault('METIS_DLL', '/usr/lib/x86_64-linux-gnu/libmetis.so')

# Detect torch version for PyG wheel URL
import torch as _torch
_pyg_whl_url = f'https://data.pyg.org/whl/torch-{_torch.__version__}.html'

def _pip_install(*packages, find_links=None):
    cmd = [sys.executable, '-m', 'pip', 'install', '-q']
    if find_links:
        cmd += ['-f', find_links]
    cmd += list(packages)
    subprocess.check_call(cmd)

# PyG ecosystem (needs matching torch/CUDA wheels)
_pip_install('torch-scatter', 'torch-sparse', 'torch-cluster', 'torch-geometric',
             find_links=_pyg_whl_url)

# Other dependencies
_pip_install('numpy', 'scipy', 'scikit-learn', 'networkx',
             'einops', 'ogb', 'yacs', 'metis', 'tensorboard')

# ── SECTION 1: Third-party imports ──────────────────────────────
import os
import re
import sys
import math
import time
import types
import random
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple

from scipy import sparse as sp

from einops.layers.torch import Rearrange

import torch_geometric
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import ZINC, TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose, Constant
from torch_scatter import scatter as _scatter_impl


@torch._dynamo.disable
def scatter(*args, **kwargs):
    """torch_scatter ops are opaque to torch.compile; run them outside Dynamo."""
    return _scatter_impl(*args, **kwargs)
from torch_sparse import SparseTensor

import networkx as nx
try:
    import metis as metis_lib
    _METIS_AVAILABLE = True
except (ImportError, RuntimeError):
    _METIS_AVAILABLE = False

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

from ogb.graphproppred.mol_encoder import AtomEncoder as AtomEncoder_
from ogb.graphproppred.mol_encoder import BondEncoder as BondEncoder_

from tqdm.auto import tqdm


# ── SECTION 2: Reproducibility seed ─────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def dataloader_num_workers(requested: int) -> int:
    """Jupyter/IPython runs user code under a transient __main__; DataLoader workers
    pickle batches and cannot resolve custom classes (e.g. SubgraphsData) defined there."""
    if requested <= 0:
        return 0
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            return 0
    except ImportError:
        pass
    return requested


def enable_h100_optimizations():
    """Enable GPU optimizations for H100 (also benefits A100/RTX 40xx)."""
    # TF32: ~2x faster float32 matmuls with negligible precision loss
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # cuDNN autotuner: finds fastest conv algorithms for fixed input sizes
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    # Set float32 matmul precision to high (uses TF32 internally)
    torch.set_float32_matmul_precision('high')


# ── SECTION 3: Per-dataset configs (inlined from YAML) ───────────

def build_cfg(dataset_name):
    """Build a SimpleNamespace config matching the YAML structure for each dataset."""

    # Shared defaults
    cfg = types.SimpleNamespace(
        dataset=dataset_name,
        num_workers=8,
        device='cuda:0',
        seed=None,
        depth=-1,
    )

    cfg.train = types.SimpleNamespace(
        batch_size=128,
        epochs=1000,
        runs=4,
        lr=0.001,
        lr_patience=20,
        lr_decay=0.5,
        wd=0.0,           # paper default: no weight decay for SSL pretraining
        dropout=0.0,      # paper default: no dropout
        mlpmixer_dropout=0.0,
        min_lr=1e-5,
        optimizer='Adam', # paper uses Adam
        multiscale=False,
        grad_clip_norm=5.0,
        ridge_alpha=1.0,
        scale_linear_eval=False,
    )

    cfg.model = types.SimpleNamespace(
        gnn_type='GINEConv',
        gMHA_type='Hadamard',
        hidden_size=512,
        nlayer_gnn=4,
        nlayer_mlpmixer=4,
        pool='mean',
        residual=True,
    )

    cfg.pos_enc = types.SimpleNamespace(
        rw_dim=0,
        lap_dim=0,
        patch_rw_dim=8,
        patch_num_diff=0,
    )

    cfg.metis = types.SimpleNamespace(
        enable=True,
        online=True,
        n_patches=32,
        drop_rate=0.3,
        num_hops=1,
    )

    cfg.jepa = types.SimpleNamespace(
        enable=True,
        num_context=1,
        num_targets=4,
        dist=0,
        num_scales=1,
        scale_factor=4,
        loss_weights=[1.0, 0.5, 0.25],
        num_targets_L1=4,
        num_targets_L2=1,
        var_weight=0.01,
    )

    # Per-dataset overrides
    if dataset_name == 'ZINC':
        cfg.train.epochs = 30
        cfg.train.batch_size = 128
        cfg.train.lr = 0.0005
        cfg.train.lr_patience = 20
        cfg.train.runs = 10
        cfg.model.nlayer_gnn = 2           # paper Table 8
        cfg.model.nlayer_mlpmixer = 4
        cfg.metis.n_patches = 32
        cfg.pos_enc.rw_dim = 20            # paper Table 8
        cfg.pos_enc.patch_rw_dim = 20
        cfg.pos_enc.lap_dim = 0            # paper does not use LapPE
        cfg.pos_enc.patch_num_diff = 0
        cfg.jepa.num_targets = 4           # paper: 1-4

    elif dataset_name == 'MUTAG':
        cfg.train.epochs = 50
        cfg.train.batch_size = 128
        cfg.train.lr = 0.0005
        cfg.train.lr_patience = 20
        cfg.train.runs = 5
        cfg.model.nlayer_gnn = 2           # paper Table 8
        cfg.model.nlayer_mlpmixer = 4
        cfg.metis.n_patches = 32
        cfg.pos_enc.rw_dim = 15            # paper Table 8
        cfg.pos_enc.patch_rw_dim = 15
        cfg.pos_enc.lap_dim = 0            # paper does not use LapPE
        cfg.pos_enc.patch_num_diff = 0
        cfg.jepa.num_targets = 3           # paper Table 8: "1-3"

    elif dataset_name == 'PROTEINS':
        cfg.train.epochs = 30
        cfg.train.batch_size = 128
        cfg.train.lr = 0.0005
        cfg.train.lr_patience = 20
        cfg.train.runs = 5
        cfg.model.nlayer_gnn = 2           # paper Table 8
        cfg.model.nlayer_mlpmixer = 4
        cfg.metis.n_patches = 32
        cfg.pos_enc.rw_dim = 20            # paper Table 8
        cfg.pos_enc.patch_rw_dim = 20
        cfg.pos_enc.lap_dim = 0            # paper does not use LapPE
        cfg.pos_enc.patch_num_diff = 0
        cfg.jepa.num_targets = 2           # paper Table 8: "1-2"

    elif dataset_name == 'DD':
        cfg.train.epochs = 20
        cfg.train.batch_size = 128
        cfg.train.lr = 0.0005
        cfg.train.lr_patience = 20
        cfg.train.runs = 5
        cfg.model.nlayer_gnn = 3           # paper Table 8
        cfg.model.nlayer_mlpmixer = 4
        cfg.metis.n_patches = 32
        cfg.pos_enc.rw_dim = 30            # paper Table 8
        cfg.pos_enc.patch_rw_dim = 30
        cfg.pos_enc.lap_dim = 0
        cfg.pos_enc.patch_num_diff = 0
        cfg.jepa.num_targets = 4           # paper Table 8: "1-4"

    elif dataset_name == 'EXP':
        cfg.train.epochs = 30
        cfg.train.batch_size = 128
        cfg.train.lr = 0.001
        cfg.train.lr_patience = 20
        cfg.train.runs = 4
        cfg.model.nlayer_gnn = 2
        cfg.model.nlayer_mlpmixer = 4
        cfg.metis.n_patches = 32
        cfg.pos_enc.rw_dim = 10
        cfg.pos_enc.patch_rw_dim = 10
        cfg.pos_enc.lap_dim = 0
        cfg.pos_enc.patch_num_diff = 0
        cfg.jepa.num_targets = 4

    elif dataset_name == 'IMDB-BINARY':
        cfg.train.epochs = 10
        cfg.train.batch_size = 32
        cfg.train.lr = 0.0005
        cfg.train.lr_patience = 30
        cfg.train.runs = 5
        cfg.model.nlayer_gnn = 2           # paper Table 8
        cfg.model.nlayer_mlpmixer = 4
        cfg.metis.n_patches = 32
        cfg.pos_enc.rw_dim = 15            # paper Table 8
        cfg.pos_enc.patch_rw_dim = 15
        cfg.pos_enc.lap_dim = 0            # paper does not use LapPE
        cfg.pos_enc.patch_num_diff = 0
        cfg.jepa.num_targets = 4           # paper Table 8: "1-4"

    elif dataset_name == 'IMDB-MULTI':
        cfg.train.epochs = 5
        cfg.train.batch_size = 32
        cfg.train.lr = 0.0005
        cfg.train.lr_patience = 20
        cfg.train.runs = 5
        cfg.model.nlayer_gnn = 2           # paper Table 8
        cfg.model.nlayer_mlpmixer = 4      # paper Table 8
        cfg.metis.n_patches = 32
        cfg.pos_enc.rw_dim = 15            # paper Table 8
        cfg.pos_enc.patch_rw_dim = 15
        cfg.pos_enc.lap_dim = 0            # paper does not use LapPE
        cfg.pos_enc.patch_num_diff = 0
        cfg.jepa.num_targets = 4           # paper Table 8: "1-4"

    elif dataset_name == 'REDDIT-BINARY':
        cfg.train.epochs = 10
        cfg.train.batch_size = 32
        cfg.train.lr = 0.0001
        cfg.train.lr_patience = 20
        cfg.train.runs = 5
        cfg.model.nlayer_gnn = 2
        cfg.model.nlayer_mlpmixer = 4
        cfg.metis.n_patches = 128
        cfg.pos_enc.rw_dim = 40
        cfg.pos_enc.patch_rw_dim = 40
        cfg.pos_enc.lap_dim = 0
        cfg.pos_enc.patch_num_diff = 0
        cfg.jepa.num_targets = 4

    elif dataset_name == 'REDDIT-MULTI-5K':
        cfg.train.epochs = 10
        cfg.train.batch_size = 32
        cfg.train.lr = 0.0001
        cfg.train.lr_patience = 20
        cfg.train.runs = 5
        cfg.model.nlayer_gnn = 2
        cfg.model.nlayer_mlpmixer = 4
        cfg.metis.n_patches = 128
        cfg.pos_enc.rw_dim = 40
        cfg.pos_enc.patch_rw_dim = 40
        cfg.pos_enc.lap_dim = 0
        cfg.pos_enc.patch_num_diff = 0
        cfg.jepa.num_targets = 4

    # Always set k for k-fold
    cfg.k = 10

    return cfg


def _maybe_inject_node_graph_pe(x: Tensor, data: Data, module: nn.Module) -> Tensor:
    """When enabled, add RWSE/LapPE MLPs to subgraph-node features (SOTA path)."""
    if not getattr(module, 'inject_node_pe', False):
        return x
    if getattr(module, 'use_rw', False):
        x = x + module.rw_encoder(data.rw_pos_enc[data.subgraphs_nodes_mapper])
    if getattr(module, 'use_lap', False):
        x = x + module.lap_encoder(data.lap_pos_enc[data.subgraphs_nodes_mapper])
    return x


def apply_sota_push_overrides(cfg, benchmark_key: str):
    """Aggressive training / architecture defaults to chase higher scores (not paper-repro).

    Toggle via main(): SOTA_PUSH = True. Benchmark key is the public name, e.g. 'ZINC', 'MUTAG'."""
    # Hierarchical multi-scale JEPA (GraphHMSJepa + GraphHMSJEPAPartitionTransform)
    cfg.jepa.num_scales = 3

    cfg.train.optimizer = 'AdamW'
    cfg.train.wd = 1e-4
    cfg.train.dropout = 0.05
    cfg.train.mlpmixer_dropout = 0.05

    # Slightly wider trunk if you still use 512 as base from build_cfg per dataset
    cfg.model.hidden_size = max(getattr(cfg.model, 'hidden_size', 512), 640)

    # LapPE only where graphs are small enough (avoid dense eig on huge graphs)
    _lap_ok = benchmark_key in (
        'ZINC', 'MUTAG', 'PROTEINS', 'EXP', 'IMDB-BINARY', 'IMDB-MULTI')
    cfg.pos_enc.lap_dim = 8 if _lap_ok else 0

    cfg.model.inject_node_pe = True

    if benchmark_key == 'ZINC':
        cfg.model.nlayer_gnn = max(cfg.model.nlayer_gnn, 3)
        cfg.train.epochs = max(cfg.train.epochs, 50)
        cfg.train.scale_linear_eval = True

    # Richer JEPA targets on TU (paper used smaller m on some sets)
    if benchmark_key in ('PROTEINS', 'MUTAG'):
        cfg.jepa.num_targets = 4


# ── SECTION 4: Positional encodings ─────────────────────────────

def random_walk(A, n_iter):
    """Geometric diffusion features with Random Walk."""
    Dinv = A.sum(dim=-1).clamp(min=1).pow(-1).unsqueeze(-1)  # D^-1
    RW = A * Dinv
    M = RW
    M_power = M
    # Iterate
    PE = [torch.diagonal(M)]
    for _ in range(n_iter - 1):
        M_power = torch.matmul(M_power, M)
        PE.append(torch.diagonal(M_power))
    PE = torch.stack(PE, dim=-1)
    return PE


def RWSE(edge_index, pos_enc_dim, num_nodes):
    """Initializing positional encoding with RWSE."""
    if edge_index.size(-1) == 0:
        PE = torch.zeros(num_nodes, pos_enc_dim)
    else:
        A = torch_geometric.utils.to_dense_adj(
            edge_index, max_num_nodes=num_nodes)[0]
        PE = random_walk(A, pos_enc_dim)
    return PE


def LapPE(edge_index, pos_enc_dim, num_nodes):
    """Graph positional encoding v/ Laplacian eigenvectors."""
    degree = torch_geometric.utils.degree(edge_index[0], num_nodes)
    A = torch_geometric.utils.to_scipy_sparse_matrix(
        edge_index, num_nodes=num_nodes)
    N = sp.diags(np.array(degree.clip(1) ** -0.5, dtype=float))
    L = sp.eye(num_nodes) - N * A * N

    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    PE = torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float()
    if PE.size(1) < pos_enc_dim:
        zeros = torch.zeros(num_nodes, pos_enc_dim)
        zeros[:, :PE.size(1)] = PE
        PE = zeros
    return PE


# ── SECTION 5: Subgraph extractors ──────────────────────────────

def k_hop_subgraph(edge_index, num_nodes, num_hops, is_directed=False):
    """Return k-hop subgraphs for all nodes in the graph."""
    if is_directed:
        row, col = edge_index
        birow, bicol = torch.cat([row, col]), torch.cat([col, row])
        edge_index = torch.stack([birow, bicol])
    else:
        row, col = edge_index
    sparse_adj = SparseTensor(
        row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
    hop_masks = [torch.eye(num_nodes, dtype=torch.bool,
                           device=edge_index.device)]
    hop_indicator = row.new_full((num_nodes, num_nodes), -1)
    hop_indicator[hop_masks[0]] = 0
    for i in range(num_hops):
        next_mask = sparse_adj.matmul(hop_masks[i].float()) > 0
        hop_masks.append(next_mask)
        hop_indicator[(hop_indicator == -1) & next_mask] = i + 1
    hop_indicator = hop_indicator.T  # N x N
    node_mask = (hop_indicator >= 0)  # N x N dense mask matrix
    return node_mask


def random_subgraph(g, n_patches, num_hops=1):
    membership = np.arange(g.num_nodes)
    np.random.shuffle(membership)
    membership = torch.tensor(membership % n_patches)
    max_patch_id = torch.max(membership) + 1
    membership = membership + (n_patches - max_patch_id)

    node_mask = torch.stack([membership == i for i in range(n_patches)])

    if num_hops > 0:
        subgraphs_batch, subgraphs_node_mapper = node_mask.nonzero().T
        k_hop_node_mask = k_hop_subgraph(
            g.edge_index, g.num_nodes, num_hops)
        node_mask[subgraphs_batch] += k_hop_node_mask[subgraphs_node_mapper]

    edge_mask = node_mask[:, g.edge_index[0]] & node_mask[:, g.edge_index[1]]
    return node_mask, edge_mask


def metis_subgraph(g, n_patches, drop_rate=0.0, num_hops=1, is_directed=False):
    if is_directed:
        if g.num_nodes < n_patches:
            membership = torch.arange(g.num_nodes)
        else:
            G = torch_geometric.utils.to_networkx(g, to_undirected="lower")
            cuts, membership = metis_lib.part_graph(G, n_patches, recursive=True)
    else:
        if g.num_nodes < n_patches:
            membership = torch.randperm(n_patches)
        else:
            adjlist = g.edge_index.t()
            arr = torch.rand(len(adjlist))
            selected = arr > drop_rate
            G = nx.Graph()
            G.add_nodes_from(np.arange(g.num_nodes))
            G.add_edges_from(adjlist[selected].tolist())
            cuts, membership = metis_lib.part_graph(G, n_patches, recursive=True)

    assert len(membership) >= g.num_nodes
    membership = torch.tensor(np.array(membership[:g.num_nodes]))
    max_patch_id = torch.max(membership) + 1
    membership = membership + (n_patches - max_patch_id)

    node_mask = torch.stack([membership == i for i in range(n_patches)])

    if num_hops > 0:
        subgraphs_batch, subgraphs_node_mapper = node_mask.nonzero().T
        k_hop_node_mask = k_hop_subgraph(
            g.edge_index, g.num_nodes, num_hops, is_directed)
        node_mask.index_add_(0, subgraphs_batch,
                             k_hop_node_mask[subgraphs_node_mapper])

    edge_mask = node_mask[:, g.edge_index[0]] & node_mask[:, g.edge_index[1]]
    return node_mask, edge_mask


# ── SECTION 6: Data transforms ──────────────────────────────────

def cal_coarsen_adj(subgraphs_nodes_mask):
    mask = subgraphs_nodes_mask.to(torch.float)
    coarsen_adj = torch.matmul(mask, mask.t())
    return coarsen_adj


def to_sparse(node_mask, edge_mask):
    subgraphs_nodes = node_mask.nonzero().T
    subgraphs_edges = edge_mask.nonzero().T
    return subgraphs_nodes, subgraphs_edges


def combine_subgraphs(edge_index, subgraphs_nodes, subgraphs_edges, num_selected=None, num_nodes=None):
    if num_selected is None:
        num_selected = subgraphs_nodes[0][-1] + 1
    if num_nodes is None:
        num_nodes = subgraphs_nodes[1].max() + 1

    combined_subgraphs = edge_index[:, subgraphs_edges[1]]
    node_label_mapper = edge_index.new_full((num_selected, num_nodes), -1)
    node_label_mapper[subgraphs_nodes[0], subgraphs_nodes[1]] = torch.arange(len(subgraphs_nodes[1]))
    node_label_mapper = node_label_mapper.reshape(-1)

    inc = torch.arange(num_selected) * num_nodes
    combined_subgraphs += inc[subgraphs_edges[0]]
    combined_subgraphs = node_label_mapper[combined_subgraphs]
    return combined_subgraphs


def combine_subgraphs_jepa(edge_index, subgraphs_nodes, subgraphs_edges,
                           context_nodes_mask, context_edges_mask,
                           target_nodes_mask, target_edges_mask,
                           num_selected=None, num_nodes=None):
    if num_selected is None:
        num_selected = subgraphs_nodes[0][-1] + 1
    if num_nodes is None:
        num_nodes = subgraphs_nodes[1].max() + 1

    combined_subgraphs = edge_index[:, subgraphs_edges[1]]
    node_label_mapper = edge_index.new_full((num_selected, num_nodes), -1)
    node_label_mapper[subgraphs_nodes[0], subgraphs_nodes[1]] = torch.arange(len(subgraphs_nodes[1]))
    node_label_mapper = node_label_mapper.reshape(-1)

    inc = torch.arange(num_selected) * num_nodes
    combined_subgraphs += inc[subgraphs_edges[0]]
    combined_subgraphs = node_label_mapper[combined_subgraphs]

    return combined_subgraphs


class SubgraphsData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        num_nodes = self.num_nodes
        num_edges = self.edge_index.size(-1)
        if bool(re.search('(combined_subgraphs)', key)):
            return getattr(self, key[:-len('combined_subgraphs')] + 'subgraphs_nodes_mapper').size(0)
        elif bool(re.search('(subgraphs_batch)', key)):
            return 1 + getattr(self, key)[-1]
        elif bool(re.search('(nodes_mapper)', key)):
            return num_nodes
        elif bool(re.search('(edges_mapper)', key)):
            return num_edges
        elif bool(re.search('(fine_to_medium)', key)):
            if hasattr(self, 'n_patches_L1'):
                v = self.n_patches_L1
                return v[0] if isinstance(v, list) else int(v)
            return 0
        elif bool(re.search('(medium_to_coarse)', key)):
            if hasattr(self, 'n_patches_L2'):
                v = self.n_patches_L2
                return v[0] if isinstance(v, list) else int(v)
            return 0
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if bool(re.search('(combined_subgraphs)', key)):
            return -1
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)


class PositionalEncodingTransform(object):
    def __init__(self, rw_dim=0, lap_dim=0):
        super().__init__()
        self.rw_dim = rw_dim
        self.lap_dim = lap_dim

    def __call__(self, data):
        if self.rw_dim > 0:
            data.rw_pos_enc = RWSE(data.edge_index, self.rw_dim, data.num_nodes)
        if self.lap_dim > 0:
            data.lap_pos_enc = LapPE(data.edge_index, self.lap_dim, data.num_nodes)
        return data


def _sample_jepa_context_target_patches(subgraph_batch_ids: Tensor, num_context: int, num_targets: int):
    """Pick distinct patch ids when possible; if the graph uses fewer patches than needed, sample with replacement."""
    pool = torch.unique(subgraph_batch_ids).detach().cpu().numpy()
    n_need = num_context + num_targets
    if pool.size == 0:
        raise RuntimeError('No patch ids in subgraph batch (empty graph?)')
    return np.random.choice(pool, n_need, replace=(pool.size < n_need))


class GraphJEPAPartitionTransform(object):
    def __init__(self, n_patches, metis=True, drop_rate=0.0, num_hops=1, is_directed=False,
                 patch_rw_dim=0, patch_num_diff=0, num_context=1, num_targets=4):
        super().__init__()
        self.n_patches = n_patches
        self.drop_rate = drop_rate
        self.num_hops = num_hops
        self.is_directed = is_directed
        self.patch_rw_dim = patch_rw_dim
        self.patch_num_diff = patch_num_diff
        self.metis = metis
        self.num_context = num_context
        self.num_targets = num_targets

    def _diffuse(self, A):
        if self.patch_num_diff == 0:
            return A
        Dinv = A.sum(dim=-1).clamp(min=1).pow(-1).unsqueeze(-1)  # D^-1
        RW = A * Dinv
        M = RW
        M_power = M
        for _ in range(self.patch_num_diff - 1):
            M_power = torch.matmul(M_power, M)
        return M_power

    def __call__(self, data):
        data = SubgraphsData(**{k: v for k, v in data})
        if self.metis:
            node_masks, edge_masks = metis_subgraph(
                data, n_patches=self.n_patches, drop_rate=self.drop_rate,
                num_hops=self.num_hops, is_directed=self.is_directed)
        else:
            node_masks, edge_masks = random_subgraph(
                data, n_patches=self.n_patches, num_hops=self.num_hops)
        subgraphs_nodes, subgraphs_edges = to_sparse(node_masks, edge_masks)

        combined_subgraphs = combine_subgraphs(
            data.edge_index, subgraphs_nodes, subgraphs_edges,
            num_selected=self.n_patches, num_nodes=data.num_nodes)

        if self.patch_num_diff > -1 or self.patch_rw_dim > 0:
            coarsen_adj = cal_coarsen_adj(node_masks)
            if self.patch_rw_dim > 0:
                data.patch_pe = random_walk(coarsen_adj, self.patch_rw_dim)
            if self.patch_num_diff > -1:
                data.coarsen_adj = self._diffuse(coarsen_adj).unsqueeze(0)

        subgraphs_batch = subgraphs_nodes[0]
        mask = torch.zeros(self.n_patches).bool()
        mask[subgraphs_batch] = True
        data.subgraphs_batch = subgraphs_batch
        data.subgraphs_nodes_mapper = subgraphs_nodes[1]
        data.subgraphs_edges_mapper = subgraphs_edges[1]
        data.combined_subgraphs = combined_subgraphs
        data.mask = mask.unsqueeze(0)

        rand_choice = _sample_jepa_context_target_patches(
            subgraphs_nodes[0], self.num_context, self.num_targets)
        context_subgraph_idx = rand_choice[0]
        target_subgraph_idxs = torch.tensor(rand_choice[1:])

        data.context_edges_mask = subgraphs_edges[0] == context_subgraph_idx
        data.target_edges_mask = torch.isin(subgraphs_edges[0], target_subgraph_idxs)
        data.context_nodes_mapper = subgraphs_nodes[1, subgraphs_nodes[0] == context_subgraph_idx]
        data.target_nodes_mapper = subgraphs_nodes[1, torch.isin(subgraphs_nodes[0], target_subgraph_idxs)]
        data.context_nodes_subgraph = subgraphs_nodes[0, subgraphs_nodes[0] == context_subgraph_idx]
        data.target_nodes_subgraph = subgraphs_nodes[0, torch.isin(subgraphs_nodes[0], target_subgraph_idxs)]
        data.context_subgraph_idx = context_subgraph_idx.tolist()
        data.target_subgraph_idxs = target_subgraph_idxs.tolist()
        data.call_n_patches = [self.n_patches]

        data.__num_nodes__ = data.num_nodes
        return data


class GraphHMSJEPAPartitionTransform(object):
    """Hierarchical Multi-Scale JEPA partition transform.

    Builds a 3-level patch hierarchy:
      L0 (fine)   : n_patches          patches  (default 32)
      L1 (medium) : n_patches_L1       patches  (default  8)
      L2 (coarse) : n_patches_L2       patches  (default  2)
    """

    def __init__(self, n_patches, scale_factor=4, metis_enable=True,
                 drop_rate=0.0, num_hops=1, is_directed=False,
                 patch_rw_dim=0, patch_num_diff=0,
                 num_context=1, num_targets=4,
                 num_targets_L1=4, num_targets_L2=1):
        super().__init__()
        self.n_patches = n_patches
        self.scale_factor = scale_factor
        self.n_patches_L1 = max(2, n_patches // scale_factor)
        self.n_patches_L2 = max(2, self.n_patches_L1 // scale_factor)
        self.drop_rate = drop_rate
        self.num_hops = num_hops
        self.is_directed = is_directed
        self.patch_rw_dim = patch_rw_dim
        self.patch_num_diff = patch_num_diff
        self.metis_enable = metis_enable
        self.num_context = num_context
        self.num_targets = num_targets
        self.num_targets_L1 = num_targets_L1
        self.num_targets_L2 = num_targets_L2

    def _diffuse(self, A):
        if self.patch_num_diff == 0:
            return A
        Dinv = A.sum(dim=-1).clamp(min=1).pow(-1).unsqueeze(-1)
        RW = A * Dinv
        M_power = RW
        for _ in range(self.patch_num_diff - 1):
            M_power = torch.matmul(M_power, RW)
        return M_power

    def _coarsen_adj(self, fine_adj, fine_to_coarse, n_coarse):
        n_fine = fine_adj.shape[0]
        M = torch.zeros(n_coarse, n_fine, dtype=fine_adj.dtype)
        M[fine_to_coarse, torch.arange(n_fine)] = 1.0
        return M @ fine_adj @ M.T

    def _metis_partition(self, adj, n_target, n_current):
        if not _METIS_AVAILABLE or n_target <= 1 or n_current <= n_target:
            membership = torch.arange(n_current) % max(1, n_target)
        else:
            G = nx.Graph()
            G.add_nodes_from(range(n_current))
            adj_np = (adj > 0).numpy()
            rows, cols = np.where(np.triu(adj_np, k=1))
            if len(rows) > 0:
                G.add_edges_from(zip(rows.tolist(), cols.tolist()))
            try:
                _, membership_list = metis_lib.part_graph(G, n_target, recursive=True)
                membership = torch.tensor(membership_list[:n_current], dtype=torch.long)
            except Exception:
                membership = torch.arange(n_current) % n_target

        max_id = membership.max().item() + 1
        membership = membership + (n_target - max_id)
        return membership

    def __call__(self, data):
        data = SubgraphsData(**{k: v for k, v in data})

        # Level 0 – fine patches
        n_L0 = self.n_patches
        if self.metis_enable:
            node_masks, edge_masks = metis_subgraph(
                data, n_patches=n_L0, drop_rate=self.drop_rate,
                num_hops=self.num_hops, is_directed=self.is_directed)
        else:
            node_masks, edge_masks = random_subgraph(
                data, n_patches=n_L0, num_hops=self.num_hops)

        subgraphs_nodes, subgraphs_edges = to_sparse(node_masks, edge_masks)
        combined_subgraphs = combine_subgraphs(
            data.edge_index, subgraphs_nodes, subgraphs_edges,
            num_selected=n_L0, num_nodes=data.num_nodes)

        coarsen_adj_L0 = cal_coarsen_adj(node_masks)

        if self.patch_rw_dim > 0:
            data.patch_pe = random_walk(coarsen_adj_L0, self.patch_rw_dim)
        if self.patch_num_diff > -1:
            data.coarsen_adj = self._diffuse(coarsen_adj_L0).unsqueeze(0)

        subgraphs_batch = subgraphs_nodes[0]
        mask_L0 = torch.zeros(n_L0).bool()
        mask_L0[subgraphs_batch] = True

        data.subgraphs_batch = subgraphs_batch
        data.subgraphs_nodes_mapper = subgraphs_nodes[1]
        data.subgraphs_edges_mapper = subgraphs_edges[1]
        data.combined_subgraphs = combined_subgraphs
        data.mask = mask_L0.unsqueeze(0)

        rand_L0 = _sample_jepa_context_target_patches(
            subgraphs_nodes[0], self.num_context, self.num_targets)
        context_subgraph_idx = rand_L0[0]
        target_subgraph_idxs = torch.tensor(rand_L0[1:])
        data.context_edges_mask = subgraphs_edges[0] == context_subgraph_idx
        data.target_edges_mask = torch.isin(subgraphs_edges[0], target_subgraph_idxs)
        data.context_nodes_mapper = subgraphs_nodes[1, subgraphs_nodes[0] == context_subgraph_idx]
        data.target_nodes_mapper = subgraphs_nodes[1, torch.isin(subgraphs_nodes[0], target_subgraph_idxs)]
        data.context_nodes_subgraph = subgraphs_nodes[0, subgraphs_nodes[0] == context_subgraph_idx]
        data.target_nodes_subgraph = subgraphs_nodes[0, torch.isin(subgraphs_nodes[0], target_subgraph_idxs)]
        data.context_subgraph_idx = context_subgraph_idx.tolist()
        data.target_subgraph_idxs = target_subgraph_idxs.tolist()
        data.call_n_patches = [n_L0]

        # Level 1 – medium patches
        n_L1 = self.n_patches_L1
        fine_to_medium = self._metis_partition(coarsen_adj_L0, n_L1, n_L0)

        coarsen_adj_L1 = self._coarsen_adj(coarsen_adj_L0, fine_to_medium, n_L1)

        if self.patch_rw_dim > 0:
            data.patch_pe_L1 = random_walk(coarsen_adj_L1, self.patch_rw_dim)
        data.coarsen_adj_L1 = self._diffuse(coarsen_adj_L1).unsqueeze(0)
        data.mask_L1 = torch.ones(n_L1, dtype=torch.bool).unsqueeze(0)
        data.fine_to_medium = fine_to_medium
        data.n_patches_L1 = [n_L1]

        all_L1 = np.arange(n_L1)
        n_sel_L1 = min(self.num_context + self.num_targets_L1, n_L1)
        rand_L1 = np.random.choice(all_L1, n_sel_L1, replace=False)
        data.context_subgraph_idx_L1 = int(rand_L1[0])
        data.target_subgraph_idxs_L1 = rand_L1[1:].tolist()

        # Level 2 – coarse patches
        n_L2 = self.n_patches_L2
        medium_to_coarse = self._metis_partition(coarsen_adj_L1, n_L2, n_L1)

        coarsen_adj_L2 = self._coarsen_adj(coarsen_adj_L1, medium_to_coarse, n_L2)

        if self.patch_rw_dim > 0:
            data.patch_pe_L2 = random_walk(coarsen_adj_L2, self.patch_rw_dim)
        data.coarsen_adj_L2 = self._diffuse(coarsen_adj_L2).unsqueeze(0)
        data.mask_L2 = torch.ones(n_L2, dtype=torch.bool).unsqueeze(0)
        data.medium_to_coarse = medium_to_coarse
        data.n_patches_L2 = [n_L2]

        all_L2 = np.arange(n_L2)
        n_sel_L2 = min(self.num_context + self.num_targets_L2, n_L2)
        rand_L2 = np.random.choice(all_L2, n_sel_L2, replace=False)
        data.context_subgraph_idx_L2 = int(rand_L2[0])
        data.target_subgraph_idxs_L2 = rand_L2[1:].tolist()

        data.__num_nodes__ = data.num_nodes
        return data


# ── SECTION 7: Dataset creation ─────────────────────────────────

class PlanarSATPairsDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(PlanarSATPairsDataset, self).__init__(
            root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["GRAPHSAT.pkl"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        pass

    def process(self):
        data_list = pickle.load(
            open(os.path.join(self.root, "raw/GRAPHSAT.pkl"), "rb"))

        data_list = [Data(**g.__dict__) for g in data_list]
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def calculate_stats(dataset):
    num_graphs = len(dataset)
    ave_num_nodes = np.array([g.num_nodes for g in dataset]).mean()
    ave_num_edges = np.array([g.num_edges for g in dataset]).mean()
    print(
        f'# Graphs: {num_graphs}, average # nodes per graph: {ave_num_nodes}, '
        f'average # edges per graph: {ave_num_edges}.')


def create_dataset(cfg):
    pre_transform = PositionalEncodingTransform(
        rw_dim=cfg.pos_enc.rw_dim, lap_dim=cfg.pos_enc.lap_dim)

    transform_train = transform_eval = None

    if cfg.metis.n_patches > 0:
        use_hms = getattr(cfg.jepa, 'num_scales', 1) > 1
        _TransformCls = GraphHMSJEPAPartitionTransform if use_hms else GraphJEPAPartitionTransform

        common_kwargs = dict(
            n_patches=cfg.metis.n_patches,
            patch_rw_dim=cfg.pos_enc.patch_rw_dim,
            patch_num_diff=cfg.pos_enc.patch_num_diff,
            num_context=cfg.jepa.num_context,
            num_targets=cfg.jepa.num_targets,
        )
        if use_hms:
            common_kwargs.update(dict(
                metis_enable=cfg.metis.enable,
                scale_factor=cfg.jepa.scale_factor,
                num_hops=cfg.metis.num_hops,
                is_directed=cfg.dataset == 'TreeDataset',
                num_targets_L1=cfg.jepa.num_targets_L1,
                num_targets_L2=cfg.jepa.num_targets_L2,
            ))
        else:
            common_kwargs.update(dict(
                metis=cfg.metis.enable,
                num_hops=cfg.metis.num_hops,
                is_directed=cfg.dataset == 'TreeDataset',
            ))

        _transform_train = _TransformCls(drop_rate=cfg.metis.drop_rate, **common_kwargs)
        _transform_eval = _TransformCls(drop_rate=0.0, **common_kwargs)
        transform_train = _transform_train
        transform_eval = _transform_eval
    else:
        print('Not supported...')
        sys.exit()

    if cfg.dataset == 'ZINC':
        root = 'dataset/ZINC'
        train_dataset = ZINC(
            root, subset=True, split='train', pre_transform=pre_transform, transform=transform_train)
        val_dataset = ZINC(root, subset=True, split='val',
                           pre_transform=pre_transform, transform=transform_eval)
        test_dataset = ZINC(root, subset=True, split='test',
                            pre_transform=pre_transform, transform=transform_eval)

    elif cfg.dataset in ['PROTEINS', 'MUTAG', 'DD', 'REDDIT-BINARY', 'REDDIT-MULTI-5K', 'IMDB-BINARY', 'IMDB-MULTI']:
        if cfg.dataset not in ['PROTEINS', 'MUTAG', 'DD']:
            pre_transform = Compose([Constant(value=0, cat=False), pre_transform])

        dataset = TUDataset(root='dataset/TUD', name=cfg.dataset, pre_transform=pre_transform)
        return dataset, transform_train, transform_eval

    elif cfg.dataset == 'exp-classify':
        root = "dataset/EXP/"
        dataset = PlanarSATPairsDataset(root, pre_transform=pre_transform)
        return dataset, transform_train, transform_eval

    else:
        print("Dataset not supported.")
        sys.exit(1)

    torch.set_num_threads(cfg.num_workers)
    if not cfg.metis.online:
        train_dataset = [x for x in train_dataset]
    val_dataset = [x for x in val_dataset]
    test_dataset = [x for x in test_dataset]

    return train_dataset, val_dataset, test_dataset


# ── SECTION 8: Model utilities ──────────────────────────────────

# ---- elements.py ----

BN = True


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input):
        return input

    def reset_parameters(self):
        pass


class MLP(nn.Module):
    def __init__(self, nin, nout, nlayer=2, with_final_activation=True, with_norm=BN, bias=True):
        super().__init__()
        n_hid = nin
        self.layers = nn.ModuleList([nn.Linear(nin if i == 0 else n_hid,
                                                n_hid if i < nlayer - 1 else nout,
                                                bias=True if (i == nlayer - 1 and not with_final_activation and bias)
                                                or (not with_norm) else False)
                                     for i in range(nlayer)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(n_hid if i < nlayer - 1 else nout) if with_norm else Identity()
                                    for i in range(nlayer)])
        self.nlayer = nlayer
        self.with_final_activation = with_final_activation
        self.residual = (nin == nout)

    def reset_parameters(self):
        for layer, norm in zip(self.layers, self.norms):
            layer.reset_parameters()
            norm.reset_parameters()

    def forward(self, x):
        previous_x = x
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            x = layer(x)
            if i < self.nlayer - 1 or self.with_final_activation:
                x = norm(x)
                x = F.relu(x)
        return x


class VNUpdate(nn.Module):
    def __init__(self, dim, with_norm=BN):
        super().__init__()
        self.mlp = MLP(dim, dim, with_norm=with_norm,
                       with_final_activation=True, bias=not BN)

    def reset_parameters(self):
        self.mlp.reset_parameters()

    def forward(self, vn, x, batch):
        from torch_geometric.nn import global_add_pool
        G = global_add_pool(x, batch)
        if vn is not None:
            G += vn
        vn = self.mlp(G)
        x += vn[batch]
        return vn, x


# ---- feature_encoder.py ----

def AtomEncoder(nin, nhid):
    return AtomEncoder_(nhid)


def BondEncoder(nin, nhid):
    return BondEncoder_(nhid)


def DiscreteEncoder(nin, nhid):
    return nn.Embedding(nin, nhid)


def LinearEncoder(nin, nhid):
    return nn.Linear(nin, nhid)


def FeatureEncoder(TYPE, nin, nhid):
    models = {
        'Atom': AtomEncoder,
        'Bond': BondEncoder,
        'Discrete': DiscreteEncoder,
        'Linear': LinearEncoder,
    }
    return models[TYPE](nin, nhid)


# ---- gnn_wrapper.py ----

class GCNConv(nn.Module):
    def __init__(self, nin, nout, bias=True):
        super().__init__()
        self.layer = pyg_nn.GCNConv(nin, nout, bias=bias)

    def reset_parameters(self):
        self.layer.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        return self.layer(x, edge_index)


class ResGatedGraphConv(nn.Module):
    def __init__(self, nin, nout, bias=True):
        super().__init__()
        self.layer = pyg_nn.ResGatedGraphConv(nin, nout, bias=bias)

    def reset_parameters(self):
        self.layer.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        return self.layer(x, edge_index)


class GINEConv(nn.Module):
    def __init__(self, nin, nout, bias=True):
        super().__init__()
        self.nn = MLP(nin, nout, 2, False, bias=bias)
        self.layer = pyg_nn.GINEConv(self.nn, train_eps=True)

    def reset_parameters(self):
        self.layer.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        return self.layer(x, edge_index, edge_attr)


class TransformerConv(nn.Module):
    def __init__(self, nin, nout, bias=True, nhead=8):
        super().__init__()
        self.layer = pyg_nn.TransformerConv(
            in_channels=nin, out_channels=nout // nhead, heads=nhead, edge_dim=nin, bias=bias)

    def reset_parameters(self):
        self.layer.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        return self.layer(x, edge_index, edge_attr)


class GATConv(nn.Module):
    def __init__(self, nin, nout, bias=True, nhead=1):
        super().__init__()
        self.layer = pyg_nn.GATConv(nin, nout // nhead, nhead, bias=bias)

    def reset_parameters(self):
        self.layer.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        return self.layer(x, edge_index, edge_attr)


class GatedGraphConv(nn.Module):
    def __init__(self, nin, nout, bias=True):
        super().__init__()
        self.layer = pyg_nn.GatedGraphConv(nin, nout, bias=bias)

    def reset_parameters(self):
        self.layer.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        return self.layer(x, edge_index)


# ---- gnn.py ----

class GNN(nn.Module):
    def __init__(self, nin, nout, nlayer_gnn, gnn_type, bn=BN, dropout=0.0, res=True):
        super().__init__()
        self.dropout = dropout
        self.res = res

        _gnn_wrapper = {
            'GCNConv': GCNConv,
            'ResGatedGraphConv': ResGatedGraphConv,
            'GINEConv': GINEConv,
            'TransformerConv': TransformerConv,
            'GATConv': GATConv,
            'GatedGraphConv': GatedGraphConv,
        }
        self.convs = nn.ModuleList([_gnn_wrapper[gnn_type](
            nin, nin, bias=not bn) for _ in range(nlayer_gnn)])
        self.norms = nn.ModuleList(
            [nn.BatchNorm1d(nin) if bn else Identity() for _ in range(nlayer_gnn)])
        self.output_encoder = nn.Linear(nin, nout)

    def forward(self, x, edge_index, edge_attr):
        previous_x = x
        for layer, norm in zip(self.convs, self.norms):
            x = layer(x, edge_index, edge_attr)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            if self.res:
                x = x + previous_x
                previous_x = x

        x = self.output_encoder(x)
        return x


# ---- mlp_mixer.py ----

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)


class MixerBlock(nn.Module):
    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout=0.):
        super().__init__()
        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b p d -> b d p'),
            FeedForward(num_patch, token_dim, dropout),
            Rearrange('b d p -> b p d'),
        )
        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout),
        )

    def forward(self, x):
        x = x + self.token_mix(x)
        x = x + self.channel_mix(x)
        return x


class MLPMixer(nn.Module):
    def __init__(self, nlayer, nhid, n_patches, with_final_norm=True, dropout=0):
        super().__init__()
        self.n_patches = n_patches
        self.with_final_norm = with_final_norm
        self.mixer_blocks = nn.ModuleList(
            [MixerBlock(nhid, self.n_patches, nhid * 4, nhid // 2, dropout=dropout) for _ in range(nlayer)])
        if self.with_final_norm:
            self.layer_norm = nn.LayerNorm(nhid)

    def forward(self, x):
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        if self.with_final_norm:
            x = self.layer_norm(x)
        return x


# ---- gMHA_hadamard.py ----

class _HadamardMultiheadAttention(nn.Module):
    """Multi-headed attention (Hadamard variant)."""

    def __init__(self, embed_dim, num_heads, kdim=None, vdim=None, dropout=0.0,
                 bias=True, self_attention=False, batch_first=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.batch_first = batch_first
        self.num_heads = num_heads
        self.dropout_module = nn.Dropout(dropout)

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention
        assert self.self_attention, "Only support self attention"
        assert not self.self_attention or self.qkv_same_dim

        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.reset_parameters()

    def prepare_for_onnx_export_(self):
        raise NotImplementedError

    def reset_parameters(self):
        if self.qkv_same_dim:
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, query, key: Optional[Tensor], value: Optional[Tensor],
                attn_bias: Optional[Tensor], key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None,
                before_softmax: bool = False, need_head_weights: bool = False,
                A: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        if need_head_weights:
            need_weights = True

        is_batched = query.dim() == 3
        if self.batch_first and is_batched:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                assert key_bsz == bsz
                assert value is not None
                assert src_len, bsz == value.shape[:2]

        q = self.q_proj(query)
        k = self.k_proj(query)
        v = self.v_proj(query)
        q *= self.scaling

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        assert k is not None
        assert k.size(1) == src_len

        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_bias is not None:
            attn_weights += attn_bias.contiguous().view(bsz * self.num_heads, tgt_len, src_len)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v

        # numerical stability
        max_val = attn_weights.max(dim=-1, keepdim=True)[0]
        attn_weights = torch.exp(attn_weights - max_val)
        attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        if A is not None:
            A = torch.repeat_interleave(A, repeats=self.num_heads, dim=0)
            attn_weights = attn_weights * A
        attn_probs = self.dropout_module(attn_weights)

        assert v is not None
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        if self.batch_first and is_batched:
            attn = attn.transpose(1, 0)

        attn_weights_out: Optional[Tensor] = None
        if need_weights:
            attn_weights_out = attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)
            if not need_head_weights:
                attn_weights_out = attn_weights_out.mean(dim=0)

        return attn, attn_weights_out

    def apply_sparse_mask(self, attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights


class HadamardEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation=F.relu, layer_norm_eps=1e-5, batch_first=False,
                 norm_first=True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(HadamardEncoderLayer, self).__init__()
        self.self_attn = _HadamardMultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout,
            self_attention=True, batch_first=batch_first)
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)
        self.num_head = nhead

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def _sa_block(self, x, attn_mask, key_padding_mask, attn_bias, A):
        x = self.self_attn(x, x, x,
                           attn_bias=attn_bias,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False,
                           A=A)[0]
        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, attn_bias=None, A=None):
        x = src
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x), src_mask, src_key_padding_mask, attn_bias=attn_bias, A=A)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(
                x + self._sa_block(x, src_mask, src_key_padding_mask, attn_bias=attn_bias, A=A))
            x = self.norm2(x + self._ff_block(x))
        return x


# ---- gMHA_gt.py ----

class _GTMultiheadAttention(nn.Module):
    """Multi-headed attention (GT/Standard variant)."""

    def __init__(self, embed_dim, num_heads, kdim=None, vdim=None, dropout=0.0,
                 bias=True, self_attention=False, batch_first=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.batch_first = batch_first
        self.num_heads = num_heads
        self.dropout_module = nn.Dropout(dropout)

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention
        assert self.self_attention, "Only support self attention"
        assert not self.self_attention or self.qkv_same_dim

        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.reset_parameters()

    def prepare_for_onnx_export_(self):
        raise NotImplementedError

    def reset_parameters(self):
        if self.qkv_same_dim:
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, query, key: Optional[Tensor], value: Optional[Tensor],
                attn_bias: Optional[Tensor], key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None,
                before_softmax: bool = False, need_head_weights: bool = False,
                A: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        if need_head_weights:
            need_weights = True

        is_batched = query.dim() == 3
        if self.batch_first and is_batched:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                assert key_bsz == bsz
                assert value is not None
                assert src_len, bsz == value.shape[:2]

        q = self.q_proj(query)
        k = self.k_proj(query)
        v = self.v_proj(query)
        q *= self.scaling

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        assert k is not None
        assert k.size(1) == src_len

        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            # assert key_padding_mask.size(1) == src_len
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_bias is not None:
            attn_weights += attn_bias.contiguous().view(bsz * self.num_heads, tgt_len, src_len)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v

        # numerical stability
        max_val = attn_weights.max(dim=-1, keepdim=True)[0]
        attn_weights = torch.exp(attn_weights - max_val)
        if A is not None:
            A = torch.repeat_interleave(A, repeats=self.num_heads, dim=0)
            attn_weights = attn_weights * A
        attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        attn_probs = self.dropout_module(attn_weights)

        assert v is not None
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        if self.batch_first and is_batched:
            attn = attn.transpose(1, 0)

        attn_weights_out: Optional[Tensor] = None
        if need_weights:
            attn_weights_out = attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)
            if not need_head_weights:
                attn_weights_out = attn_weights_out.mean(dim=0)

        return attn, attn_weights_out

    def apply_sparse_mask(self, attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights


class GTEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation=F.relu, layer_norm_eps=1e-5, batch_first=False,
                 norm_first=True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(GTEncoderLayer, self).__init__()
        self.self_attn = _GTMultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout,
            self_attention=True, batch_first=batch_first)
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)
        self.num_head = nhead

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def _sa_block(self, x, attn_mask, key_padding_mask, attn_bias, A):
        x = self.self_attn(x, x, x,
                           attn_bias=attn_bias,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False,
                           A=A)[0]
        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, attn_bias=None, A=None):
        x = src
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x), src_mask, src_key_padding_mask, attn_bias=attn_bias, A=A)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(
                x + self._sa_block(x, src_mask, src_key_padding_mask, attn_bias=attn_bias, A=A))
            x = self.norm2(x + self._ff_block(x))
        return x


# ---- gMHA_graphormer.py ----

class _GraphormerMultiheadAttention(nn.Module):
    """Multi-headed attention (Graphormer variant, no A kwarg in forward)."""

    def __init__(self, embed_dim, num_heads, kdim=None, vdim=None, dropout=0.0,
                 bias=True, self_attention=False, batch_first=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.batch_first = batch_first
        self.num_heads = num_heads
        self.dropout_module = nn.Dropout(dropout)

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention
        assert self.self_attention, "Only support self attention"
        assert not self.self_attention or self.qkv_same_dim

        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.reset_parameters()

    def prepare_for_onnx_export_(self):
        raise NotImplementedError

    def reset_parameters(self):
        if self.qkv_same_dim:
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, query, key: Optional[Tensor], value: Optional[Tensor],
                attn_bias: Optional[Tensor], key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None,
                before_softmax: bool = False,
                need_head_weights: bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        if need_head_weights:
            need_weights = True

        is_batched = query.dim() == 3
        if self.batch_first and is_batched:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                assert key_bsz == bsz
                assert value is not None
                assert src_len, bsz == value.shape[:2]

        q = self.q_proj(query)
        k = self.k_proj(query)
        v = self.v_proj(query)
        q *= self.scaling

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        assert k is not None
        assert k.size(1) == src_len

        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_bias is not None:
            attn_weights += attn_bias.contiguous().view(bsz * self.num_heads, tgt_len, src_len)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v

        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout_module(attn_weights)

        assert v is not None
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        if self.batch_first and is_batched:
            attn = attn.transpose(1, 0)

        attn_weights_out: Optional[Tensor] = None
        if need_weights:
            attn_weights_out = attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)
            if not need_head_weights:
                attn_weights_out = attn_weights_out.mean(dim=0)

        return attn, attn_weights_out

    def apply_sparse_mask(self, attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights


class GraphormerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation=F.relu, layer_norm_eps=1e-5, batch_first=False,
                 norm_first=True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(GraphormerEncoderLayer, self).__init__()
        self.self_attn = _GraphormerMultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout,
            self_attention=True, batch_first=batch_first)
        self.spatial_pos_encoder = nn.Linear(1, nhead)
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)
        self.num_head = nhead

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def forward(self, src, src_mask=None, src_key_padding_mask=None, A=None):
        x = src
        attn_bias = None
        if A is not None:
            attn_bias = self.spatial_pos_encoder(A.unsqueeze(-1)).permute(0, 3, 1, 2)
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, attn_bias=attn_bias)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, attn_bias=attn_bias))
            x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(self, x, attn_mask, key_padding_mask, attn_bias):
        x = self.self_attn(x, x, x,
                           attn_bias=attn_bias,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


# ---- gMHA_wrapper.py ----

class _gMHA_MLPMixer(nn.Module):
    def __init__(self, nhid, dropout, nlayer, n_patches, with_final_norm=True):
        super().__init__()
        self.n_patches = n_patches
        self.with_final_norm = with_final_norm
        self.mixer_blocks = nn.ModuleList(
            [MixerBlock(nhid, self.n_patches, nhid * 4, nhid // 2, dropout=dropout) for _ in range(nlayer)])
        if self.with_final_norm:
            self.layer_norm = nn.LayerNorm(nhid)

    def forward(self, x, coarsen_adj, mask):
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        if self.with_final_norm:
            x = self.layer_norm(x)
        return x


class _gMHA_Hadamard(nn.Module):
    """Hadamard attention (default): (A ⊙ softmax(QK^T/sqrt(d)))V"""

    def __init__(self, nhid, dropout, nlayer, n_patches, nhead=8, batch_first=True):
        super().__init__()
        self.transformer_encoder = nn.ModuleList([HadamardEncoderLayer(
            d_model=nhid, dim_feedforward=nhid * 2, nhead=nhead,
            batch_first=batch_first, dropout=dropout) for _ in range(nlayer)])

    def forward(self, x, coarsen_adj, mask):
        for layer in self.transformer_encoder:
            x = layer(x, A=coarsen_adj, src_key_padding_mask=mask)
        return x


class _gMHA_Standard(nn.Module):
    """Standard (full) attention: softmax(QK^T/sqrt(d))V"""

    def __init__(self, nhid, dropout, nlayer, n_patches, nhead=8, batch_first=True):
        super().__init__()
        self.transformer_encoder = nn.ModuleList([GTEncoderLayer(
            d_model=nhid, dim_feedforward=nhid * 2, nhead=nhead,
            batch_first=batch_first, dropout=dropout) for _ in range(nlayer)])

    def forward(self, x, coarsen_adj, mask):
        for layer in self.transformer_encoder:
            x = layer(x, A=None, src_key_padding_mask=mask)
        return x


class _gMHA_Graph(nn.Module):
    """Graph attention (GT-like): softmax(A ⊙ QK^T/sqrt(d))V"""

    def __init__(self, nhid, dropout, nlayer, n_patches, nhead=8, batch_first=True):
        super().__init__()
        self.transformer_encoder = nn.ModuleList([GTEncoderLayer(
            d_model=nhid, dim_feedforward=nhid * 2, nhead=nhead,
            batch_first=batch_first, dropout=dropout) for _ in range(nlayer)])

    def forward(self, x, coarsen_adj, mask):
        for layer in self.transformer_encoder:
            x = layer(x, A=coarsen_adj, src_key_padding_mask=mask)
        return x


class _gMHA_Kernel(nn.Module):
    """Kernel attention (GraphiT-like)."""

    def __init__(self, nhid, dropout, nlayer, n_patches, nhead=8, batch_first=True):
        super().__init__()
        self.transformer_encoder = nn.ModuleList([GTEncoderLayer(
            d_model=nhid, dim_feedforward=nhid * 2, nhead=nhead,
            batch_first=batch_first, dropout=dropout) for _ in range(nlayer)])

    def forward(self, x, coarsen_adj_dense, mask):
        for layer in self.transformer_encoder:
            x = layer(x, A=coarsen_adj_dense, src_key_padding_mask=mask)
        return x


class _gMHA_Addictive(nn.Module):
    """Addictive attention (Graphormer-like): softmax(QK^T/sqrt(d))V + LL(A)"""

    def __init__(self, nhid, dropout, nlayer, n_patches, nhead=8, batch_first=True):
        super().__init__()
        self.transformer_encoder = nn.ModuleList([GraphormerEncoderLayer(
            d_model=nhid, dim_feedforward=nhid * 2, nhead=nhead,
            batch_first=batch_first, dropout=dropout) for _ in range(nlayer)])

    def forward(self, x, coarsen_adj_dense, mask):
        for layer in self.transformer_encoder:
            x = layer(x, A=coarsen_adj_dense, src_key_padding_mask=mask)
        return x


# Namespace object to replicate module-attribute access pattern used in model.py
class _gMHA_wrapper_ns:
    MLPMixer = _gMHA_MLPMixer
    Hadamard = _gMHA_Hadamard
    Standard = _gMHA_Standard
    Graph = _gMHA_Graph
    Kernel = _gMHA_Kernel
    Addictive = _gMHA_Addictive


gMHA_wrapper = _gMHA_wrapper_ns()


# ── SECTION 9: Core models (GraphJepa, GraphHMSJepa) ─────────────

class GraphJepa(nn.Module):

    def __init__(self,
                 nfeat_node, nfeat_edge,
                 nhid, nout,
                 nlayer_gnn,
                 nlayer_mlpmixer,
                 node_type, edge_type,
                 gnn_type,
                 gMHA_type='MLPMixer',
                 rw_dim=0,
                 lap_dim=0,
                 dropout=0,
                 mlpmixer_dropout=0,
                 bn=True,
                 res=True,
                 pooling='mean',
                 n_patches=32,
                 patch_rw_dim=0,
                 num_context_patches=1,
                 num_target_patches=4,
                 inject_node_pe=False):

        super().__init__()
        self.dropout = dropout
        self.inject_node_pe = inject_node_pe
        self.use_rw = rw_dim > 0
        self.use_lap = lap_dim > 0
        self.n_patches = n_patches
        self.pooling = pooling
        self.res = res
        self.patch_rw_dim = patch_rw_dim
        self.nhid = nhid
        self.nfeat_edge = nfeat_edge
        self.num_context_patches = num_context_patches
        self.num_target_patches = num_target_patches

        if self.use_rw:
            self.rw_encoder = MLP(rw_dim, nhid, 1)
        if self.use_lap:
            self.lap_encoder = MLP(lap_dim, nhid, 1)
        if self.patch_rw_dim > 0:
            self.patch_rw_encoder = MLP(self.patch_rw_dim, nhid, 1)

        self.input_encoder = FeatureEncoder(node_type, nfeat_node, nhid)
        self.edge_encoder = FeatureEncoder(edge_type, nfeat_edge, nhid)

        self.gnns = nn.ModuleList([GNN(nin=nhid, nout=nhid, nlayer_gnn=1, gnn_type=gnn_type,
                                       bn=bn, dropout=dropout, res=res) for _ in range(nlayer_gnn)])
        self.U = nn.ModuleList(
            [MLP(nhid, nhid, nlayer=1, with_final_activation=True) for _ in range(nlayer_gnn - 1)])

        self.reshape = Rearrange('(B p) d ->  B p d', p=n_patches)

        self.context_encoder = getattr(gMHA_wrapper, 'Standard')(
            nhid=nhid, dropout=mlpmixer_dropout, nlayer=nlayer_mlpmixer, n_patches=n_patches)
        self.target_encoder = getattr(gMHA_wrapper, gMHA_type)(
            nhid=nhid, dropout=mlpmixer_dropout, nlayer=nlayer_mlpmixer, n_patches=n_patches)

        # Predictor
        self.target_predictor = MLP(
            nhid, 2, nlayer=3, with_final_activation=False, with_norm=False)

    def forward(self, data):
        x = self.input_encoder(data.x).squeeze()

        edge_attr = data.edge_attr
        if edge_attr is None:
            edge_attr = data.edge_index.new_zeros(data.edge_index.size(-1)).float().unsqueeze(-1)
        edge_attr = self.edge_encoder(edge_attr)

        # Patch Encoder
        x = x[data.subgraphs_nodes_mapper]
        x = _maybe_inject_node_graph_pe(x, data, self)
        edge_index = data.combined_subgraphs
        e = edge_attr[data.subgraphs_edges_mapper]
        batch_x = data.subgraphs_batch
        pes = data.rw_pos_enc[data.subgraphs_nodes_mapper]
        patch_pes = scatter(pes, batch_x, dim=0, reduce='max')
        for i, gnn in enumerate(self.gnns):
            if i > 0:
                subgraph = scatter(x, batch_x, dim=0, reduce=self.pooling)[batch_x]
                x = x + self.U[i - 1](subgraph)
                x = scatter(x, data.subgraphs_nodes_mapper, dim=0, reduce='mean')[data.subgraphs_nodes_mapper]
            x = gnn(x, edge_index, e)
        subgraph_x = scatter(x, batch_x, dim=0, reduce=self.pooling)

        ######################## Graph-JEPA ########################
        batch_indexer = torch.tensor(np.cumsum(data.call_n_patches))
        batch_indexer = torch.hstack((torch.tensor(0), batch_indexer[:-1])).to(data.y.device)

        context_subgraph_idx = data.context_subgraph_idx + batch_indexer
        target_subgraphs_idx = torch.vstack([torch.tensor(dt) for dt in data.target_subgraph_idxs]).to(data.y.device)
        target_subgraphs_idx += batch_indexer.unsqueeze(1)

        context_subgraphs = subgraph_x[context_subgraph_idx]
        target_subgraphs = subgraph_x[target_subgraphs_idx.flatten()]

        target_pes = patch_pes[target_subgraphs_idx.flatten()]
        context_pe = patch_pes[context_subgraph_idx]
        context_subgraphs += self.patch_rw_encoder(context_pe)
        encoded_tpatch_pes = self.patch_rw_encoder(target_pes)

        target_x = target_subgraphs.reshape(-1, self.num_target_patches, self.nhid)
        context_x = context_subgraphs.unsqueeze(1)

        context_mask = data.mask.flatten()[context_subgraph_idx].reshape(-1, self.num_context_patches)
        context_x = self.context_encoder(context_x, data.coarsen_adj if hasattr(
            data, 'coarsen_adj') else None, ~context_mask)

        with torch.no_grad():
            if hasattr(data, 'coarsen_adj'):
                subgraph_incides = torch.vstack([torch.tensor(dt) for dt in data.target_subgraph_idxs])
                patch_adj = data.coarsen_adj[
                    torch.arange(target_x.shape[0]).unsqueeze(1).unsqueeze(2),
                    subgraph_incides.unsqueeze(1),
                    subgraph_incides.unsqueeze(2)
                ]
                target_x = self.target_encoder(target_x, patch_adj, None)
            else:
                target_x = self.target_encoder(target_x, None, None)

            # Predict coordinates in the Q1 hyperbola
            x_coord = torch.cosh(target_x.mean(-1).unsqueeze(-1))
            y_coord = torch.sinh(target_x.mean(-1).unsqueeze(-1))
            target_x = torch.cat([x_coord, y_coord], dim=-1)

        target_prediction_embeddings = context_x + encoded_tpatch_pes.reshape(-1, self.num_target_patches, self.nhid)
        target_y = self.target_predictor(target_prediction_embeddings)

        return target_x, target_y

    def encode(self, data):
        x = self.input_encoder(data.x).squeeze()

        edge_attr = data.edge_attr
        if edge_attr is None:
            edge_attr = data.edge_index.new_zeros(data.edge_index.size(-1)).float().unsqueeze(-1)
        edge_attr = self.edge_encoder(edge_attr)

        # Patch Encoder
        x = x[data.subgraphs_nodes_mapper]
        x = _maybe_inject_node_graph_pe(x, data, self)
        edge_index = data.combined_subgraphs
        e = edge_attr[data.subgraphs_edges_mapper]
        batch_x = data.subgraphs_batch
        pes = data.rw_pos_enc[data.subgraphs_nodes_mapper]
        patch_pes = scatter(pes, batch_x, dim=0, reduce='mean')
        for i, gnn in enumerate(self.gnns):
            if i > 0:
                subgraph = scatter(x, batch_x, dim=0, reduce=self.pooling)[batch_x]
                x = x + self.U[i - 1](subgraph)
                x = scatter(x, data.subgraphs_nodes_mapper, dim=0, reduce='mean')[data.subgraphs_nodes_mapper]
            x = gnn(x, edge_index, e)
        subgraph_x = scatter(x, batch_x, dim=0, reduce=self.pooling)
        subgraph_x += self.patch_rw_encoder(patch_pes)

        # Handles different patch sizes based on the data object for multiscale training
        mixer_x = subgraph_x.reshape(len(data.call_n_patches), data.call_n_patches[0][0], -1)

        # Eval via target encoder
        mixer_x = self.target_encoder(mixer_x, data.coarsen_adj if hasattr(
            data, 'coarsen_adj') else None, ~data.mask)

        # Global Average Pooling
        out = (mixer_x * data.mask.unsqueeze(-1)).sum(1) / data.mask.sum(1, keepdim=True)
        return out

    def encode_nopool(self, data):
        x = self.input_encoder(data.x).squeeze()

        edge_attr = data.edge_attr
        if edge_attr is None:
            edge_attr = data.edge_index.new_zeros(data.edge_index.size(-1)).float().unsqueeze(-1)
        edge_attr = self.edge_encoder(edge_attr)

        # Patch Encoder
        x = x[data.subgraphs_nodes_mapper]
        x = _maybe_inject_node_graph_pe(x, data, self)
        edge_index = data.combined_subgraphs
        e = edge_attr[data.subgraphs_edges_mapper]
        batch_x = data.subgraphs_batch
        pes = data.rw_pos_enc[data.subgraphs_nodes_mapper]
        patch_pes = scatter(pes, batch_x, dim=0, reduce='mean')
        for i, gnn in enumerate(self.gnns):
            if i > 0:
                subgraph = scatter(x, batch_x, dim=0, reduce=self.pooling)[batch_x]
                x = x + self.U[i - 1](subgraph)
                x = scatter(x, data.subgraphs_nodes_mapper, dim=0, reduce='mean')[data.subgraphs_nodes_mapper]
            x = gnn(x, edge_index, e)
        subgraph_x = scatter(x, batch_x, dim=0, reduce=self.pooling)
        subgraph_x += self.patch_rw_encoder(patch_pes)

        # Eval via target encoder
        mixer_x = subgraph_x.reshape(len(data.call_n_patches), data.call_n_patches[0], -1)
        mixer_x = self.target_encoder(mixer_x, data.coarsen_adj if hasattr(
            data, 'coarsen_adj') else None, ~data.mask)

        return mixer_x


class GraphHMSJepa(nn.Module):
    """Graph JEPA with 3-level hierarchical predictive coding.

    Three JEPA objectives are computed simultaneously:
      1. Same-scale  (L0 → L0): fine context predicts fine targets
      2. Cross-scale (L0 → L1): fine context predicts medium targets
      3. Cross-scale (L1 → L2): medium context predicts coarse targets
    """

    def __init__(self,
                 nfeat_node, nfeat_edge,
                 nhid, nout,
                 nlayer_gnn,
                 nlayer_mlpmixer,
                 node_type, edge_type,
                 gnn_type,
                 gMHA_type='Hadamard',
                 rw_dim=0,
                 lap_dim=0,
                 dropout=0,
                 mlpmixer_dropout=0,
                 bn=True,
                 res=True,
                 pooling='mean',
                 n_patches=32,
                 patch_rw_dim=0,
                 num_context_patches=1,
                 num_target_patches=4,
                 num_target_patches_L1=4,
                 num_target_patches_L2=1,
                 loss_weights=(1.0, 0.5, 0.25),
                 var_weight=0.01,
                 inject_node_pe=False):

        super().__init__()
        self.dropout = dropout
        self.inject_node_pe = inject_node_pe
        self.use_rw = rw_dim > 0
        self.use_lap = lap_dim > 0
        self.n_patches = n_patches
        self.pooling = pooling
        self.res = res
        self.patch_rw_dim = patch_rw_dim
        self.nhid = nhid
        self.nfeat_edge = nfeat_edge
        self.num_context_patches = num_context_patches
        self.num_target_patches = num_target_patches
        self.num_target_patches_L1 = num_target_patches_L1
        self.num_target_patches_L2 = num_target_patches_L2
        self.loss_weights = list(loss_weights)
        self.var_weight = var_weight
        self.n_scales = 3

        if self.use_rw:
            self.rw_encoder = MLP(rw_dim, nhid, 1)
        if self.use_lap:
            self.lap_encoder = MLP(lap_dim, nhid, 1)

        # Patch positional encoders – one per scale
        if patch_rw_dim > 0:
            self.patch_rw_encoder = MLP(patch_rw_dim, nhid, 1)
            self.patch_rw_encoder_L1 = MLP(patch_rw_dim, nhid, 1)
            self.patch_rw_encoder_L2 = MLP(patch_rw_dim, nhid, 1)

        self.input_encoder = FeatureEncoder(node_type, nfeat_node, nhid)
        self.edge_encoder = FeatureEncoder(edge_type, nfeat_edge, nhid)

        # Shared GNN backbone
        self.gnns = nn.ModuleList([GNN(nin=nhid, nout=nhid, nlayer_gnn=1, gnn_type=gnn_type,
                                       bn=bn, dropout=dropout, res=res) for _ in range(nlayer_gnn)])
        self.U = nn.ModuleList(
            [MLP(nhid, nhid, nlayer=1, with_final_activation=True) for _ in range(nlayer_gnn - 1)])

        # Three pairs of (context_encoder, target_encoder)
        self.context_encoder_L0 = getattr(gMHA_wrapper, 'Standard')(
            nhid=nhid, dropout=mlpmixer_dropout, nlayer=nlayer_mlpmixer, n_patches=n_patches)
        self.target_encoder_L0 = getattr(gMHA_wrapper, gMHA_type)(
            nhid=nhid, dropout=mlpmixer_dropout, nlayer=nlayer_mlpmixer, n_patches=n_patches)

        self.context_encoder_L1 = getattr(gMHA_wrapper, 'Standard')(
            nhid=nhid, dropout=mlpmixer_dropout, nlayer=nlayer_mlpmixer, n_patches=n_patches)
        self.target_encoder_L1 = getattr(gMHA_wrapper, 'Hadamard')(
            nhid=nhid, dropout=mlpmixer_dropout, nlayer=nlayer_mlpmixer, n_patches=n_patches)

        self.context_encoder_L2 = getattr(gMHA_wrapper, 'Standard')(
            nhid=nhid, dropout=mlpmixer_dropout, nlayer=nlayer_mlpmixer, n_patches=n_patches)
        self.target_encoder_L2 = getattr(gMHA_wrapper, 'Hadamard')(
            nhid=nhid, dropout=mlpmixer_dropout, nlayer=nlayer_mlpmixer, n_patches=n_patches)

        # Convenience list for EMA updates in the training loop
        self.encoder_pairs = [
            (self.context_encoder_L0, self.target_encoder_L0),
            (self.context_encoder_L1, self.target_encoder_L1),
            (self.context_encoder_L2, self.target_encoder_L2),
        ]

        # Full-dimensional predictors
        self.predictor_L0_to_L0 = MLP(nhid, nhid, nlayer=3, with_final_activation=False, with_norm=False)
        self.predictor_L0_to_L1 = MLP(nhid, nhid, nlayer=2, with_final_activation=False, with_norm=False)
        self.predictor_L1_to_L2 = MLP(nhid, nhid, nlayer=2, with_final_activation=False, with_norm=False)

    def _gnn_forward(self, data):
        """Run node feature encoding + GNN; return (subgraph_x_L0, raw_patch_pes, batch_x)."""
        x = self.input_encoder(data.x).squeeze()

        edge_attr = data.edge_attr
        if edge_attr is None:
            edge_attr = data.edge_index.new_zeros(data.edge_index.size(-1)).float().unsqueeze(-1)
        edge_attr = self.edge_encoder(edge_attr)

        x = x[data.subgraphs_nodes_mapper]
        x = _maybe_inject_node_graph_pe(x, data, self)
        edge_index = data.combined_subgraphs
        e = edge_attr[data.subgraphs_edges_mapper]
        batch_x = data.subgraphs_batch
        pes = data.rw_pos_enc[data.subgraphs_nodes_mapper]
        raw_patch_pes = scatter(pes, batch_x, dim=0, reduce='max')

        for i, gnn in enumerate(self.gnns):
            if i > 0:
                subgraph = scatter(x, batch_x, dim=0, reduce=self.pooling)[batch_x]
                x = x + self.U[i - 1](subgraph)
                x = scatter(x, data.subgraphs_nodes_mapper, dim=0, reduce='mean')[data.subgraphs_nodes_mapper]
            x = gnn(x, edge_index, e)

        subgraph_x_L0 = scatter(x, batch_x, dim=0, reduce=self.pooling)
        return subgraph_x_L0, raw_patch_pes, batch_x

    def forward(self, data):
        device = data.y.device
        subgraph_x_L0, raw_patch_pes, _ = self._gnn_forward(data)

        # Hierarchical pooling
        subgraph_x_L1 = scatter(subgraph_x_L0, data.fine_to_medium, dim=0, reduce='mean')
        raw_patch_pes_L1 = scatter(raw_patch_pes, data.fine_to_medium, dim=0, reduce='mean')
        subgraph_x_L2 = scatter(subgraph_x_L1, data.medium_to_coarse, dim=0, reduce='mean')
        raw_patch_pes_L2 = scatter(raw_patch_pes_L1, data.medium_to_coarse, dim=0, reduce='mean')

        B = len(data.call_n_patches)

        def _batch_indexer(n_patches_list):
            bi = torch.tensor(np.cumsum(n_patches_list))
            return torch.hstack((torch.tensor(0), bi[:-1])).to(device)

        bi_L0 = _batch_indexer(data.call_n_patches)
        bi_L1 = _batch_indexer(data.n_patches_L1)
        bi_L2 = _batch_indexer(data.n_patches_L2)

        # L0 same-scale JEPA
        ctx_idx_L0 = data.context_subgraph_idx + bi_L0
        tgt_idx_L0 = torch.vstack([torch.tensor(dt) for dt in data.target_subgraph_idxs]).to(device)
        tgt_idx_L0 += bi_L0.unsqueeze(1)

        ctx_patch_L0 = subgraph_x_L0[ctx_idx_L0]
        tgt_patch_L0 = subgraph_x_L0[tgt_idx_L0.flatten()]
        ctx_pe_L0 = raw_patch_pes[ctx_idx_L0]
        tgt_pe_L0 = raw_patch_pes[tgt_idx_L0.flatten()]

        ctx_patch_L0 = ctx_patch_L0 + self.patch_rw_encoder(ctx_pe_L0)
        encoded_tgt_pe_L0 = self.patch_rw_encoder(tgt_pe_L0)

        ctx_x_L0 = ctx_patch_L0.unsqueeze(1)
        tgt_x_L0 = tgt_patch_L0.reshape(B, self.num_target_patches, self.nhid)

        ctx_mask_L0 = data.mask.flatten()[ctx_idx_L0].reshape(B, self.num_context_patches)
        ctx_x_L0 = self.context_encoder_L0(ctx_x_L0,
                                            data.coarsen_adj if hasattr(data, 'coarsen_adj') else None,
                                            ~ctx_mask_L0)

        with torch.no_grad():
            if hasattr(data, 'coarsen_adj'):
                tgt_rel_L0 = torch.vstack([torch.tensor(dt) for dt in data.target_subgraph_idxs])
                patch_adj_L0 = data.coarsen_adj[
                    torch.arange(B).unsqueeze(1).unsqueeze(2),
                    tgt_rel_L0.unsqueeze(1),
                    tgt_rel_L0.unsqueeze(2)]
                tgt_x_L0 = self.target_encoder_L0(tgt_x_L0, patch_adj_L0, None)
            else:
                tgt_x_L0 = self.target_encoder_L0(tgt_x_L0, None, None)

        pred_L0 = self.predictor_L0_to_L0(
            ctx_x_L0 + encoded_tgt_pe_L0.reshape(B, self.num_target_patches, self.nhid))

        # L0 → L1 cross-scale JEPA
        tgt_idx_L1 = torch.vstack([torch.tensor(dt) for dt in data.target_subgraph_idxs_L1]).to(device)
        tgt_idx_L1 += bi_L1.unsqueeze(1)
        nT_L1 = tgt_idx_L1.shape[1]

        tgt_patch_L1 = subgraph_x_L1[tgt_idx_L1.flatten()]
        tgt_pe_L1 = raw_patch_pes_L1[tgt_idx_L1.flatten()]
        encoded_tgt_pe_L1 = self.patch_rw_encoder_L1(tgt_pe_L1)

        tgt_x_L1 = tgt_patch_L1.reshape(B, nT_L1, self.nhid)

        with torch.no_grad():
            tgt_rel_L1 = torch.vstack([torch.tensor(dt) for dt in data.target_subgraph_idxs_L1])
            patch_adj_L1 = data.coarsen_adj_L1[
                torch.arange(B).unsqueeze(1).unsqueeze(2),
                tgt_rel_L1.unsqueeze(1),
                tgt_rel_L1.unsqueeze(2)]
            tgt_x_L1 = self.target_encoder_L1(tgt_x_L1, patch_adj_L1, None)

        pred_L1 = self.predictor_L0_to_L1(
            ctx_x_L0 + encoded_tgt_pe_L1.reshape(B, nT_L1, self.nhid))

        # L1 → L2 cross-scale JEPA
        ctx_idx_L1 = data.context_subgraph_idx_L1 + bi_L1
        ctx_patch_L1 = subgraph_x_L1[ctx_idx_L1]
        ctx_pe_L1 = raw_patch_pes_L1[ctx_idx_L1]
        ctx_patch_L1 = ctx_patch_L1 + self.patch_rw_encoder_L1(ctx_pe_L1)

        ctx_x_L1 = ctx_patch_L1.unsqueeze(1)
        ctx_mask_L1 = data.mask_L1.flatten()[ctx_idx_L1].reshape(B, 1)
        ctx_x_L1 = self.context_encoder_L1(ctx_x_L1, None, ~ctx_mask_L1)

        tgt_idx_L2 = torch.vstack([torch.tensor(dt) for dt in data.target_subgraph_idxs_L2]).to(device)
        tgt_idx_L2 += bi_L2.unsqueeze(1)
        nT_L2 = tgt_idx_L2.shape[1]

        tgt_patch_L2 = subgraph_x_L2[tgt_idx_L2.flatten()]
        tgt_pe_L2 = raw_patch_pes_L2[tgt_idx_L2.flatten()]
        encoded_tgt_pe_L2 = self.patch_rw_encoder_L2(tgt_pe_L2)

        tgt_x_L2 = tgt_patch_L2.reshape(B, nT_L2, self.nhid)

        with torch.no_grad():
            tgt_rel_L2 = torch.vstack([torch.tensor(dt) for dt in data.target_subgraph_idxs_L2])
            patch_adj_L2 = data.coarsen_adj_L2[
                torch.arange(B).unsqueeze(1).unsqueeze(2),
                tgt_rel_L2.unsqueeze(1),
                tgt_rel_L2.unsqueeze(2)]
            tgt_x_L2 = self.target_encoder_L2(tgt_x_L2, patch_adj_L2, None)

        pred_L2 = self.predictor_L1_to_L2(
            ctx_x_L1 + encoded_tgt_pe_L2.reshape(B, nT_L2, self.nhid))

        return (tgt_x_L0, pred_L0), (tgt_x_L1, pred_L1), (tgt_x_L2, pred_L2)

    def encode(self, data):
        subgraph_x_L0, raw_patch_pes, _ = self._gnn_forward(data)

        subgraph_x_L1 = scatter(subgraph_x_L0, data.fine_to_medium, dim=0, reduce='mean')
        raw_patch_pes_L1 = scatter(raw_patch_pes, data.fine_to_medium, dim=0, reduce='mean')
        subgraph_x_L2 = scatter(subgraph_x_L1, data.medium_to_coarse, dim=0, reduce='mean')
        raw_patch_pes_L2 = scatter(raw_patch_pes_L1, data.medium_to_coarse, dim=0, reduce='mean')

        # Add patch PEs
        subgraph_x_L0 = subgraph_x_L0 + self.patch_rw_encoder(raw_patch_pes)
        subgraph_x_L1 = subgraph_x_L1 + self.patch_rw_encoder_L1(raw_patch_pes_L1)
        subgraph_x_L2 = subgraph_x_L2 + self.patch_rw_encoder_L2(raw_patch_pes_L2)

        B = len(data.call_n_patches)
        n_L0 = data.call_n_patches[0][0]
        n_L1 = data.n_patches_L1[0][0]
        n_L2 = data.n_patches_L2[0][0]

        mixer_L0 = subgraph_x_L0.reshape(B, n_L0, -1)
        mixer_L1 = subgraph_x_L1.reshape(B, n_L1, -1)
        mixer_L2 = subgraph_x_L2.reshape(B, n_L2, -1)

        mixer_L0 = self.target_encoder_L0(mixer_L0,
                                           data.coarsen_adj if hasattr(data, 'coarsen_adj') else None,
                                           ~data.mask)
        mixer_L1 = self.target_encoder_L1(mixer_L1, data.coarsen_adj_L1, ~data.mask_L1)
        mixer_L2 = self.target_encoder_L2(mixer_L2, data.coarsen_adj_L2, ~data.mask_L2)

        out_L0 = (mixer_L0 * data.mask.unsqueeze(-1)).sum(1) / data.mask.sum(1, keepdim=True)
        out_L1 = mixer_L1.mean(1)
        out_L2 = mixer_L2.mean(1)

        return torch.cat([out_L0, out_L1, out_L2], dim=-1)    # [B, 3*nhid]


# ── SECTION 10: Model factory ─────────────────────────────────────

def create_model(cfg):
    if cfg.dataset == 'ZINC':
        node_type = 'Discrete'
        edge_type = 'Discrete'
        nfeat_node = 28
        nfeat_edge = 4
        nout = 1

    elif cfg.dataset == 'exp-classify':
        nfeat_node = 2
        nfeat_edge = 1
        node_type = 'Discrete'
        edge_type = 'Linear'
        nout = 2

    elif cfg.dataset == 'MUTAG':
        nfeat_node = 7
        nfeat_edge = 4
        node_type = 'Linear'
        edge_type = 'Linear'
        nout = 2

    elif cfg.dataset == 'PROTEINS':
        nfeat_node = 3
        nfeat_edge = 1
        node_type = 'Linear'
        edge_type = 'Linear'
        nout = 2

    elif cfg.dataset == 'DD':
        nfeat_node = 89
        nfeat_edge = 1
        node_type = 'Linear'
        edge_type = 'Linear'
        nout = 2

    elif cfg.dataset == 'REDDIT-BINARY':
        nfeat_node = 1
        nfeat_edge = 1
        node_type = 'Linear'
        edge_type = 'Linear'
        nout = 2

    elif cfg.dataset == 'REDDIT-MULTI-5K':
        nfeat_node = 1
        nfeat_edge = 1
        node_type = 'Linear'
        edge_type = 'Linear'
        nout = 5

    elif cfg.dataset == 'IMDB-BINARY':
        nfeat_node = 1
        nfeat_edge = 1
        node_type = 'Linear'
        edge_type = 'Linear'
        nout = 2

    elif cfg.dataset == 'IMDB-MULTI':
        nfeat_node = 1
        nfeat_edge = 1
        node_type = 'Linear'
        edge_type = 'Linear'
        nout = 3

    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")

    if cfg.metis.n_patches > 0:
        if cfg.jepa.enable:
            common_kwargs = dict(
                nfeat_node=nfeat_node,
                nfeat_edge=nfeat_edge,
                nhid=cfg.model.hidden_size,
                nout=nout,
                nlayer_gnn=cfg.model.nlayer_gnn,
                node_type=node_type,
                edge_type=edge_type,
                nlayer_mlpmixer=cfg.model.nlayer_mlpmixer,
                gMHA_type=cfg.model.gMHA_type,
                gnn_type=cfg.model.gnn_type,
                rw_dim=cfg.pos_enc.rw_dim,
                lap_dim=cfg.pos_enc.lap_dim,
                pooling=cfg.model.pool,
                dropout=cfg.train.dropout,
                mlpmixer_dropout=cfg.train.mlpmixer_dropout,
                n_patches=cfg.metis.n_patches,
                patch_rw_dim=cfg.pos_enc.patch_rw_dim,
                num_context_patches=cfg.jepa.num_context,
                num_target_patches=cfg.jepa.num_targets,
                inject_node_pe=getattr(cfg.model, 'inject_node_pe', False),
            )
            use_hms = getattr(cfg.jepa, 'num_scales', 1) > 1
            if use_hms:
                return GraphHMSJepa(
                    **common_kwargs,
                    num_target_patches_L1=cfg.jepa.num_targets_L1,
                    num_target_patches_L2=cfg.jepa.num_targets_L2,
                    loss_weights=cfg.jepa.loss_weights,
                    var_weight=cfg.jepa.var_weight,
                )
            else:
                return GraphJepa(**common_kwargs)
        else:
            print('Not supported...')
            sys.exit()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_optimizer(cfg, model):
    opt_name = getattr(cfg.train, 'optimizer', 'AdamW')
    wd = getattr(cfg.train, 'wd', 0.0)
    if opt_name == 'AdamW':
        return torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=wd)
    return torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=wd)


# ── SECTION 11: Loss & EMA utilities ─────────────────────────────

def hyperbolic_dist(gt, target):
    bs, _, _ = gt.size()
    gt, target = gt.view(bs, -1), target.view(bs, -1)
    nom = 2 * (torch.linalg.norm(gt - target, dim=1) ** 2)
    denom = ((1 - torch.linalg.norm(gt, dim=1) ** 2) * (1 - torch.linalg.norm(target, dim=1) ** 2))
    hdist = torch.acosh(1. + nom / denom)
    return hdist.mean()


def _unwrap_compile(model):
    """torch.compile wraps nn.Module; real class is on _orig_mod."""
    return getattr(model, '_orig_mod', model)


def _compute_loss(model, data, criterion, criterion_type):
    """Compute loss for both single-scale (GraphJepa) and HMS models."""
    if isinstance(_unwrap_compile(model), GraphHMSJepa):
        (tx0, ty0), (tx1, ty1), (tx2, ty2) = model(data)
        if criterion_type == 0:
            l0 = criterion(tx0, ty0)
            l1 = criterion(tx1, ty1)
            l2 = criterion(tx2, ty2)
        elif criterion_type == 1:
            l0 = F.mse_loss(tx0, ty0)
            l1 = F.mse_loss(tx1, ty1)
            l2 = F.mse_loss(tx2, ty2)
        else:
            print('Loss function not supported! Exiting!')
            sys.exit()
        _m = _unwrap_compile(model)
        w = _m.loss_weights
        loss = w[0] * l0 + w[1] * l1 + w[2] * l2
        loss = loss + _m.var_weight * torch.mean(
            torch.relu(1.0 - tx0.detach().std(dim=0)))
        num_t = len(ty0)
    else:
        target_x, target_y = model(data)
        if criterion_type == 0:
            loss = criterion(target_x, target_y)
        elif criterion_type == 1:
            loss = F.mse_loss(target_x, target_y)
        elif criterion_type == 2:
            loss = hyperbolic_dist(target_x, target_y)
        else:
            print('Loss function not supported! Exiting!')
            sys.exit()
        num_t = len(target_y)
    return loss, num_t


def _ema_update(model, momentum_weight):
    """Apply EMA update to all target encoders."""
    m = _unwrap_compile(model)
    if isinstance(m, GraphHMSJepa):
        for ctx_enc, tgt_enc in m.encoder_pairs:
            for param_q, param_k in zip(ctx_enc.parameters(), tgt_enc.parameters()):
                param_k.data.mul_(momentum_weight).add_((1. - momentum_weight) * param_q.detach().data)
    else:
        for param_q, param_k in zip(m.context_encoder.parameters(), m.target_encoder.parameters()):
            param_k.data.mul_(momentum_weight).add_((1. - momentum_weight) * param_q.detach().data)


# ── SECTION 12: Train / test functions (3 variants) ──────────────

# ---- Variant A: SmoothL1Loss(beta=0.5) ----
# Used by: ZINC, DD, IMDB-BINARY, IMDB-MULTI, REDDIT-BINARY, REDDIT-MULTI-5K

def train_beta05(train_loader, model, optimizer, evaluator=None, device='cpu',
                 momentum_weight=0.996, sharp=None, criterion_type=0, scaler=None,
                 grad_clip_norm=0.0):
    criterion = torch.nn.SmoothL1Loss(beta=0.5)
    step_losses, num_targets = [], []
    use_amp = scaler is not None
    for data in tqdm(train_loader, desc='Train', leave=False):
        if model.use_lap:
            batch_pos_enc = data.lap_pos_enc
            sign_flip = torch.rand(batch_pos_enc.size(1))
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            data.lap_pos_enc = batch_pos_enc * sign_flip.unsqueeze(0)
        data = data.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=use_amp):
            loss, num_t = _compute_loss(model, data, criterion, criterion_type)
        step_losses.append(loss.item())
        num_targets.append(num_t)

        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip_norm and grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        with torch.no_grad():
            _ema_update(model, momentum_weight)

    epoch_loss = np.average(step_losses, weights=num_targets)
    return None, epoch_loss


@torch.no_grad()
def test_beta05(loader, model, evaluator=None, device='cpu', criterion_type=0, scaler=None):
    criterion = torch.nn.SmoothL1Loss(beta=0.5)
    use_amp = scaler is not None
    step_losses, num_targets = [], []
    for data in tqdm(loader, desc='Eval', leave=False):
        data = data.to(device)
        with torch.amp.autocast('cuda', enabled=use_amp):
            loss, num_t = _compute_loss(model, data, criterion, criterion_type)
        step_losses.append(loss.item())
        num_targets.append(num_t)

    epoch_loss = np.average(step_losses, weights=num_targets)
    return None, epoch_loss


# ---- Variant B: SmoothL1Loss() (default beta=1.0) ----
# Used by: MUTAG, PROTEINS

def train_beta10(train_loader, model, optimizer, evaluator=None, device='cpu',
                 momentum_weight=0.996, sharp=None, criterion_type=0, scaler=None,
                 grad_clip_norm=0.0):
    criterion = torch.nn.SmoothL1Loss()
    step_losses, num_targets = [], []
    use_amp = scaler is not None
    for data in tqdm(train_loader, desc='Train', leave=False):
        if model.use_lap:
            batch_pos_enc = data.lap_pos_enc
            sign_flip = torch.rand(batch_pos_enc.size(1))
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            data.lap_pos_enc = batch_pos_enc * sign_flip.unsqueeze(0)
        data = data.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=use_amp):
            loss, num_t = _compute_loss(model, data, criterion, criterion_type)
        step_losses.append(loss.item())
        num_targets.append(num_t)

        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip_norm and grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        with torch.no_grad():
            _ema_update(model, momentum_weight)

    epoch_loss = np.average(step_losses, weights=num_targets)
    return None, epoch_loss


@torch.no_grad()
def test_beta10(loader, model, evaluator=None, device='cpu', criterion_type=0, scaler=None):
    criterion = torch.nn.SmoothL1Loss()
    use_amp = scaler is not None
    step_losses, num_targets = [], []
    for data in tqdm(loader, desc='Eval', leave=False):
        data = data.to(device)
        with torch.amp.autocast('cuda', enabled=use_amp):
            loss, num_t = _compute_loss(model, data, criterion, criterion_type)
        step_losses.append(loss.item())
        num_targets.append(num_t)

    epoch_loss = np.average(step_losses, weights=num_targets)
    return None, epoch_loss


# ---- Variant C: EXP (same loss/EMA as other datasets; supports GraphHMSJepa if enabled) ----

def train_exp(train_loader, model, optimizer, evaluator=None, device='cpu',
              momentum_weight=0.996, sharp=None, criterion_type=0, scaler=None,
              grad_clip_norm=0.0):
    criterion = torch.nn.SmoothL1Loss()
    step_losses, num_targets = [], []
    use_amp = scaler is not None
    for data in tqdm(train_loader, desc='Train', leave=False):
        if model.use_lap:
            batch_pos_enc = data.lap_pos_enc
            sign_flip = torch.rand(batch_pos_enc.size(1))
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            data.lap_pos_enc = batch_pos_enc * sign_flip.unsqueeze(0)
        data = data.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=use_amp):
            loss, num_t = _compute_loss(model, data, criterion, criterion_type)

        step_losses.append(loss.item())
        num_targets.append(num_t)

        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip_norm and grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        with torch.no_grad():
            _ema_update(model, momentum_weight)

    epoch_loss = np.average(step_losses, weights=num_targets)
    return None, epoch_loss


@torch.no_grad()
def test_exp(loader, model, evaluator=None, device='cpu', criterion_type=0, scaler=None):
    criterion = torch.nn.SmoothL1Loss()
    use_amp = scaler is not None
    step_losses, num_targets = [], []
    for data in tqdm(loader, desc='Eval', leave=False):
        data = data.to(device)
        with torch.amp.autocast('cuda', enabled=use_amp):
            loss, num_t = _compute_loss(model, data, criterion, criterion_type)

        step_losses.append(loss.item())
        num_targets.append(num_t)

    epoch_loss = np.average(step_losses, weights=num_targets)
    return None, epoch_loss


# ── SECTION 13: Training loops (run, run_k_fold) ─────────────────

def k_fold(dataset, folds=10):
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)
    train_indices, test_indices = [], []
    ys = dataset._data.y
    if torch.is_tensor(ys):
        ys = ys.view(-1).cpu().numpy()
    else:
        ys = np.asarray(ys).reshape(-1)
    for train, test in skf.split(torch.zeros(len(dataset)), ys):
        train_indices.append(torch.from_numpy(train).to(torch.long))
        test_indices.append(torch.from_numpy(test).to(torch.long))
    return train_indices, test_indices


def run(cfg, create_dataset_fn, create_model_fn, train_fn, test_fn, evaluator=None):
    if cfg.seed is not None:
        seeds = [cfg.seed]
        cfg.train.runs = 1
    else:
        seeds = [21, 42, 41, 95, 12, 35, 66, 85, 3, 1234]

    # Dummy config_logger replacement
    print(f"[INFO] Starting run for dataset: {cfg.dataset}")

    train_dataset, val_dataset, test_dataset = create_dataset_fn(cfg)

    _use_cuda = 'cuda' in cfg.device
    _nw = dataloader_num_workers(cfg.num_workers)
    _dl_kwargs = dict(num_workers=_nw,
                      pin_memory=_use_cuda,
                      persistent_workers=_nw > 0)
    train_loader = DataLoader(
        train_dataset, cfg.train.batch_size, shuffle=True, **_dl_kwargs)
    val_loader = DataLoader(
        val_dataset, cfg.train.batch_size, shuffle=False, **_dl_kwargs)
    test_loader = DataLoader(
        test_dataset, cfg.train.batch_size, shuffle=False, **_dl_kwargs)

    scaler = torch.amp.GradScaler('cuda') if _use_cuda else None

    train_losses = []
    per_epoch_times = []
    total_times = []
    maes = []
    for run_idx in range(cfg.train.runs):
        set_seed(seeds[run_idx])
        model = create_model_fn(cfg).to(cfg.device)
        if _use_cuda:
            model = torch.compile(model)
        print(f"\nNumber of parameters: {count_parameters(model)}")

        sharp = False
        optimizer = build_optimizer(cfg, model)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                               factor=cfg.train.lr_decay,
                                                               patience=cfg.train.lr_patience)
        _gclip = getattr(cfg.train, 'grad_clip_norm', 0.0)

        start_outer = time.time()
        per_epoch_time = []

        ipe = len(train_loader)
        ema_params = [0.996, 1.0]
        momentum_scheduler = (ema_params[0] + i * (ema_params[1] - ema_params[0]) / (ipe * cfg.train.epochs)
                              for i in range(int(ipe * cfg.train.epochs) + 1))
        epoch_pbar = tqdm(range(cfg.train.epochs), desc=f'Run {run_idx}')
        for epoch in epoch_pbar:
            start = time.time()
            model.train()
            _, train_loss = train_fn(
                train_loader, model, optimizer,
                evaluator=evaluator, device=cfg.device, momentum_weight=next(momentum_scheduler),
                sharp=sharp, criterion_type=cfg.jepa.dist, scaler=scaler,
                grad_clip_norm=_gclip)
            model.eval()
            _, val_loss = test_fn(val_loader, model,
                                  evaluator=evaluator, device=cfg.device, criterion_type=cfg.jepa.dist, scaler=scaler)
            _, test_loss = test_fn(test_loader, model,
                                   evaluator=evaluator, device=cfg.device, criterion_type=cfg.jepa.dist, scaler=scaler)

            time_cur_epoch = time.time() - start
            per_epoch_time.append(time_cur_epoch)

            epoch_pbar.set_postfix(train=f'{train_loss:.4f}', val=f'{val_loss:.4f}',
                                   test=f'{test_loss:.4f}', sec=f'{time_cur_epoch:.1f}')

            if scheduler is not None:
                scheduler.step(val_loss)

            if not sharp:
                if optimizer.param_groups[0]['lr'] < cfg.train.min_lr:
                    print("!! LR EQUAL TO MIN LR SET.")
                    break

        per_epoch_time = np.mean(per_epoch_time)
        total_time = (time.time() - start_outer) / 3600

        model.eval()
        X_train, y_train = [], []
        X_test, y_test = [], []
        for data in train_loader:
            data.to(cfg.device)
            with torch.no_grad():
                features = model.encode(data)
                X_train.append(features.detach().cpu().numpy())
                y_train.append(data.y.detach().cpu().numpy())

        X_train = np.concatenate(X_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)

        for data in test_loader:
            data.to(cfg.device)
            with torch.no_grad():
                features = model.encode(data)
                X_test.append(features.detach().cpu().numpy())
                y_test.append(data.y.detach().cpu().numpy())

        X_test = np.concatenate(X_test, axis=0)
        y_test = np.concatenate(y_test, axis=0)

        print("Data shapes:", X_train.shape, y_train.shape, X_test.shape, y_test.shape)

        if getattr(cfg.train, 'scale_linear_eval', False):
            _scaler = StandardScaler()
            X_train = _scaler.fit_transform(X_train)
            X_test = _scaler.transform(X_test)

        lin_model = Ridge(alpha=getattr(cfg.train, 'ridge_alpha', 1.0))
        lin_model.fit(X_train, y_train)
        lin_predictions = lin_model.predict(X_test)
        lin_mae = mean_absolute_error(y_test, lin_predictions)
        maes.append(lin_mae)

        print("\nRun: ", run_idx)
        print("Train Loss: {:.4f}".format(train_loss))
        print("Convergence Time (Epochs): {}".format(epoch + 1))
        print("AVG TIME PER EPOCH: {:.4f} s".format(per_epoch_time))
        print("TOTAL TIME TAKEN: {:.4f} h".format(total_time))
        print(f'Train R2.: {lin_model.score(X_train, y_train)}')
        print(f'MAE.: {lin_mae}')

        train_losses.append(train_loss)
        per_epoch_times.append(per_epoch_time)
        total_times.append(total_time)

    if cfg.train.runs > 1:
        train_loss_t = torch.tensor(train_losses)
        per_epoch_time_t = torch.tensor(per_epoch_times)
        total_time_t = torch.tensor(total_times)
        print(f'\nFinal Train Loss: {train_loss_t.mean():.4f} \u00b1 {train_loss_t.std():.4f}'
              f'\nSeconds/epoch: {per_epoch_time_t.mean():.4f}'
              f'\nHours/total: {total_time_t.mean():.4f}')
        maes = np.array(maes)
        print(f'MAE avg: {maes.mean()}, std: {maes.std()}')


def run_k_fold(cfg, create_dataset_fn, create_model_fn, train_fn, test_fn, evaluator=None, k=10):
    if cfg.seed is not None:
        seeds = [cfg.seed]
        cfg.train.runs = 1
    else:
        seeds = [42, 21, 95, 12, 35]

    # Dummy config_logger replacement
    print(f"[INFO] Starting k-fold run for dataset: {cfg.dataset}")

    dataset, transform, transform_eval = create_dataset_fn(cfg)

    if hasattr(dataset, 'train_indices'):
        k_fold_indices = dataset.train_indices, dataset.test_indices
    else:
        k_fold_indices = k_fold(dataset, cfg.k)

    _use_cuda = 'cuda' in cfg.device
    _nw = dataloader_num_workers(cfg.num_workers)
    _dl_kwargs = dict(num_workers=_nw,
                      pin_memory=_use_cuda,
                      persistent_workers=_nw > 0)
    scaler = torch.amp.GradScaler('cuda') if _use_cuda else None

    train_losses = []
    per_epoch_times = []
    total_times = []
    run_metrics = []
    for run_idx in range(cfg.train.runs):
        set_seed(seeds[run_idx])
        acc = []
        for fold, (train_idx, test_idx) in enumerate(zip(*k_fold_indices)):
            train_dataset = dataset[train_idx]
            test_dataset = dataset[test_idx]
            train_dataset.transform = transform
            test_dataset.transform = transform_eval
            test_dataset = [x for x in test_dataset]

            if not cfg.metis.online:
                train_dataset = [x for x in train_dataset]

            train_loader = DataLoader(
                train_dataset, cfg.train.batch_size, shuffle=True, **_dl_kwargs)
            test_loader = DataLoader(
                test_dataset, cfg.train.batch_size, shuffle=False, **_dl_kwargs)

            model = create_model_fn(cfg).to(cfg.device)
            if _use_cuda:
                model = torch.compile(model)

            optimizer = build_optimizer(cfg, model)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                   factor=cfg.train.lr_decay,
                                                                   patience=cfg.train.lr_patience)
            _gclip = getattr(cfg.train, 'grad_clip_norm', 0.0)

            start_outer = time.time()
            per_epoch_time = []

            ipe = len(train_loader)
            ema_params = [0.996, 1.0]
            momentum_scheduler = (ema_params[0] + i * (ema_params[1] - ema_params[0]) / (ipe * cfg.train.epochs)
                                  for i in range(int(ipe * cfg.train.epochs) + 1))

            epoch_pbar = tqdm(range(cfg.train.epochs), desc=f'Fold {fold}')
            for epoch in epoch_pbar:
                start = time.time()
                model.train()
                _, train_loss = train_fn(
                    train_loader, model, optimizer,
                    evaluator=evaluator, device=cfg.device,
                    momentum_weight=next(momentum_scheduler), criterion_type=cfg.jepa.dist,
                    scaler=scaler, grad_clip_norm=_gclip)
                model.eval()
                _, test_loss = test_fn(
                    test_loader, model, evaluator=evaluator, device=cfg.device,
                    criterion_type=cfg.jepa.dist, scaler=scaler)

                scheduler.step(test_loss)
                time_cur_epoch = time.time() - start
                per_epoch_time.append(time_cur_epoch)

                epoch_pbar.set_postfix(train=f'{train_loss:.4f}', test=f'{test_loss:.4f}',
                                       sec=f'{time_cur_epoch:.1f}')

                if optimizer.param_groups[0]['lr'] < cfg.train.min_lr:
                    print("!! LR EQUAL TO MIN LR SET.")
                    break

            per_epoch_time = np.mean(per_epoch_time)
            total_time = (time.time() - start_outer) / 3600

            model.eval()
            X_train, y_train = [], []
            X_test, y_test = [], []

            for data in train_loader:
                data.to(cfg.device)
                with torch.no_grad():
                    features = model.encode(data)
                    X_train.append(features.detach().cpu().numpy())
                    y_train.append(data.y.detach().cpu().numpy())

            X_train = np.concatenate(X_train, axis=0)
            y_train = np.concatenate(y_train, axis=0)

            for data in test_loader:
                data.to(cfg.device)
                with torch.no_grad():
                    features = model.encode(data)
                    X_test.append(features.detach().cpu().numpy())
                    y_test.append(data.y.detach().cpu().numpy())

            X_test = np.concatenate(X_test, axis=0)
            y_test = np.concatenate(y_test, axis=0)

            print("Data shapes:", X_train.shape, y_train.shape, X_test.shape, y_test.shape)

            x_scaler = StandardScaler()
            X_train = x_scaler.fit_transform(X_train)
            X_test = x_scaler.transform(X_test)

            lin_model = LogisticRegression(max_iter=50000)
            lin_model.fit(X_train, y_train)
            lin_predictions = lin_model.predict(X_test)
            lin_accuracy = accuracy_score(y_test, lin_predictions)
            acc.append(lin_accuracy)

            print(f'Fold {fold}, Seconds/epoch: {per_epoch_time}')
            print(f'Acc.: {lin_accuracy}')
            train_losses.append(train_loss)
            per_epoch_times.append(per_epoch_time)
            total_times.append(total_time)

        print("\nRun: ", run_idx)
        print("Train Loss: {:.4f}".format(train_loss))
        print("Convergence Time (Epochs): {}".format(epoch + 1))
        print("AVG TIME PER EPOCH: {:.4f} s".format(per_epoch_time))
        print("TOTAL TIME TAKEN: {:.4f} h".format(total_time))
        acc = np.array(acc)
        print(f'Acc mean: {acc.mean()}, std: {acc.std()}')
        run_metrics.append([acc.mean(), acc.std()])
        print()

    if cfg.train.runs > 1:
        train_loss_t = torch.tensor(train_losses)
        per_epoch_time_t = torch.tensor(per_epoch_times)
        total_time_t = torch.tensor(total_times)

        print(f'\nFinal Train Loss: {train_loss_t.mean():.4f} \u00b1 {train_loss_t.std():.4f}'
              f'\nSeconds/epoch: {per_epoch_time_t.mean():.4f}'
              f'\nHours/total: {total_time_t.mean():.4f}')

    run_metrics = np.array(run_metrics)
    print('Averages over 5 runs:')
    print(run_metrics[:, 0].mean(), run_metrics[:, 1].mean())
    print()


# ── SECTION 14: Main entry point ─────────────────────────────────

DATASET_ROUTING = {
    'ZINC':            ('ZINC',            'run',       'beta05'),
    'MUTAG':           ('MUTAG',           'run_kfold', 'beta10'),
    'PROTEINS':        ('PROTEINS',        'run_kfold', 'beta10'),
    'DD':              ('DD',              'run_kfold', 'beta05'),
    'EXP':             ('exp-classify',    'run_kfold', 'exp'),
    'IMDB-BINARY':     ('IMDB-BINARY',     'run_kfold', 'beta05'),
    'IMDB-MULTI':      ('IMDB-MULTI',      'run_kfold', 'beta05'),
    'REDDIT-BINARY':   ('REDDIT-BINARY',   'run_kfold', 'beta05'),
    'REDDIT-MULTI-5K': ('REDDIT-MULTI-5K', 'run_kfold', 'beta05'),
}

# Stable order for DATASET == 'ALL' (matches dict insertion order on Py3.7+)
ALL_BENCHMARK_KEYS = tuple(DATASET_ROUTING.keys())


def run_single_benchmark(dataset_key, device=0, seed=None, runs=None, sota_push=False):
    """Run one benchmark from DATASET_ROUTING (public name key, e.g. 'ZINC', 'MUTAG')."""
    if dataset_key not in DATASET_ROUTING:
        raise ValueError(f"Unknown dataset {dataset_key!r}. Choose from {list(DATASET_ROUTING)} or 'ALL'.")

    internal_name, runner, variant = DATASET_ROUTING[dataset_key]
    cfg = build_cfg(dataset_key)
    if sota_push:
        apply_sota_push_overrides(cfg, dataset_key)
    cfg.dataset = internal_name
    cfg.device = f'cuda:{device}' if torch.cuda.is_available() and device != 'cpu' else 'cpu'
    if 'cuda' in cfg.device:
        enable_h100_optimizations()
    if seed is not None:
        cfg.seed = seed
    if runs is not None:
        cfg.train.runs = runs

    train_fn = {'beta05': train_beta05, 'beta10': train_beta10, 'exp': train_exp}[variant]
    test_fn = {'beta05': test_beta05, 'beta10': test_beta10, 'exp': test_exp}[variant]

    if runner == 'run':
        run(cfg, create_dataset, create_model, train_fn, test_fn)
    else:
        run_k_fold(cfg, create_dataset, create_model, train_fn, test_fn, k=10)


def main():
    # ╔═══════════════════════════════════════════════════════════╗
    # ║  CONFIGURE HERE — change these variables to run          ║
    # ╚═══════════════════════════════════════════════════════════╝
    DATASET = 'ALL'         # 'ALL' = every benchmark below; or one of: ZINC, MUTAG, PROTEINS, DD,
                            # EXP, IMDB-BINARY, IMDB-MULTI, REDDIT-BINARY, REDDIT-MULTI-5K
    START_FROM = None       # Only used when DATASET='ALL'. Example: 'PROTEINS' to resume from that dataset.
    SOTA_PUSH = True       # True = HMS-JEPA + AdamW/wd/dropout + wider hidden + optional LapPE (see apply_sota_push_overrides)
    DEVICE  = 0             # GPU index (0, 1, ...) or 'cpu'
    SEED    = None          # Set an int for single deterministic run, None for paper defaults
    RUNS    = None          # Override number of runs, None for paper defaults

    if DATASET == 'ALL':
        all_keys = list(ALL_BENCHMARK_KEYS)
        if START_FROM is not None:
            if START_FROM not in all_keys:
                raise ValueError(f"START_FROM={START_FROM!r} not in benchmark keys: {all_keys}")
            all_keys = all_keys[all_keys.index(START_FROM):]
        for i, key in enumerate(all_keys):
            print('\n' + '=' * 72)
            print(f' BENCHMARK {i + 1}/{len(all_keys)}: {key}')
            print('=' * 72 + '\n')
            run_single_benchmark(key, device=DEVICE, seed=SEED, runs=RUNS, sota_push=SOTA_PUSH)
        print('\n' + '=' * 72)
        print(' ALL BENCHMARKS FINISHED')
        print('=' * 72 + '\n')
    else:
        run_single_benchmark(DATASET, device=DEVICE, seed=SEED, runs=RUNS, sota_push=SOTA_PUSH)


if __name__ == '__main__':
    main()
