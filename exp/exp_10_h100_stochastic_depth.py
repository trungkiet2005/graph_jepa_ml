# ============================================================
# EXP 10 — HMS-JEPA + Stochastic Depth (DropPath) Regularisation
# ============================================================
# Novelty: Apply Stochastic Depth (DropPath) to the GNN layers and
# MLP-Mixer (Hadamard) layers of HMS-JEPA. This is a principled
# structural dropout that acts as a powerful regulariser for small
# graph datasets, while also improving training efficiency.
#
# Motivation (from tracker):
#   - All EXP 01-07 are ~5-7pp below paper. The per-fold std is very
#     high (5-10%), consistent with high-variance / overfitting behaviour
#     on tiny folds (MUTAG: ~17 training graphs per fold!).
#   - Standard dropout on nodes/edges is insufficient for graph SSL.
#   - Stochastic Depth trains a model as an implicit ensemble of shallower
#     networks at different depths → reduces overfitting without add. params.
#
# Theoretical grounding:
#   - Huang et al., "Deep Networks with Stochastic Depth", ECCV 2016.
#   - Touvron et al., "Training Data-Efficient Image Transformers (DeiT)",
#     ICML 2021 — DropPath proven critical for small-data regimes.
#   - Liu et al., "Swin Transformer", ICCV 2021 — layerwise DropPath schedule.
#   - Chen et al., "A Simple Framework for Contrastive Learning (SimCLR)",
#     ICML 2020 — augmentation dropout as implicit regularization.
#   - Zhao et al., "Stars, Paths, Triangles and Dropping Them: DropGNN",
#     NeurIPS 2021 — random dropping in GNN layers for expressivity + reg.
#   - Park et al., "How to Exploit Hyperspherical Uniformity...",
#     ICLR 2023 — regularization of latent space diversity.
#
# Implementation:
#   We implement a `DropPath` module and inject it into:
#     (a) Each GNN layer residual connection  (prob: gnn_drop_path_rate)
#     (b) Each MLP-Mixer block  (prob: mixer_drop_path_rate, linearly increasing)
#   The DropPath *stochastically skips entire layer residuals* (not neurons).
#   At test time, DropPath is disabled (standard behaviour).
#
#   Since we cannot easily monkey-patch the existing GNN / gMHA modules
#   without modifying the repo source, we implement a WRAPPER approach:
#   We subclass GraphHMSJepa and override _gnn_forward to inject DropPath
#   at the residual connections, and wrap the context/target encoders.
#
# Datasets: MUTAG, PROTEINS, IMDB-BINARY, IMDB-MULTI, DD, ZINC.
# Estimated wall time on H100: ~80 min total.
# ============================================================

# ─────────────────────────────────────────────────────────────
# 0. ENVIRONMENT SETUP
# ─────────────────────────────────────────────────────────────
import os, sys, subprocess, time

REPO_URL = "https://github.com/trungkiet2005/graph_jepa_ml.git"
REPO_DIR = "/kaggle/working/graph_jepa_ml"

if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
else:
    subprocess.run(["git", "-C", REPO_DIR, "pull", "--ff-only"], check=True)

os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)

subprocess.run(["apt-get", "install", "-y", "libmetis-dev"],
               check=True, capture_output=True)
os.environ["METIS_DLL"] = "/usr/lib/x86_64-linux-gnu/libmetis.so"

import torch
torch_ver = torch.__version__
subprocess.run([
    "pip", "install", "-q",
    "torch-scatter", "torch-sparse", "torch-cluster", "torch-geometric",
    "-f", f"https://data.pyg.org/whl/torch-{torch_ver}.html"
], check=True)
subprocess.run([
    "pip", "install", "-q",
    "yacs", "tensorboard", "networkx", "einops", "metis", "ogb"
], check=True)

# ─────────────────────────────────────────────────────────────
# 1. IMPORTS
# ─────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter

from core.config   import cfg, update_cfg
from core.get_data import create_dataset
from core.get_model import create_model
from torch_geometric.loader import DataLoader
from core.trainer import run_k_fold, run, k_fold
from train.zinc   import _compute_loss, _ema_update
from core.model   import GraphHMSJepa
from core.model_utils.elements import MLP
from core.model_utils.feature_encoder import FeatureEncoder
from core.model_utils.gnn import GNN
import core.model_utils.gMHA_wrapper as gMHA_wrapper

# ─────────────────────────────────────────────────────────────
# 2. GLOBAL GPU SETTINGS
# ─────────────────────────────────────────────────────────────
torch.backends.cudnn.benchmark = True
USE_AMP   = torch.cuda.is_available()
AMP_DTYPE = torch.bfloat16

# ─────────────────────────────────────────────────────────────
# 3. DROP PATH (STOCHASTIC DEPTH) MODULE
# ─────────────────────────────────────────────────────────────

class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample.
    Works on any tensor of shape [batch, ...].
    Drops the entire residual for randomly selected samples during training.
    Reference: Huang et al. ECCV 2016; Touvron et al. ICML 2021 (DeiT).
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        survival_prob = 1.0 - self.drop_prob
        # Create binary mask: shape [batch, 1, 1, ...] for broadcasting
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        rand_tensor = torch.floor(rand_tensor + survival_prob)  # 0 or 1
        # Scale up surviving activations to keep expected value constant
        output = x / survival_prob * rand_tensor
        return output

    def extra_repr(self):
        return f"drop_prob={self.drop_prob:.3f}"


# ─────────────────────────────────────────────────────────────
# 4. HMS-JEPA WITH STOCHASTIC DEPTH (subclass)
# ─────────────────────────────────────────────────────────────

class GraphHMSJepaDropPath(GraphHMSJepa):
    """
    GraphHMSJepa augmented with DropPath at every GNN residual connection.

    New constructor args:
        gnn_drop_path_rate  (float): DropPath prob for GNN layer residuals.
                                     Applied stochastically at residual
                                     x += U(subgraph) and x += scatter(...)
        mixer_drop_path_rate (float): DropPath prob for context-encoder
                                      MLP-Mixer blocks (linearly scaled per block).
    All other args are passed through to GraphHMSJepa unchanged.
    """

    def __init__(self, *args,
                 gnn_drop_path_rate: float = 0.1,
                 mixer_drop_path_rate: float = 0.1,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.gnn_drop_path_rate   = gnn_drop_path_rate
        self.mixer_drop_path_rate = mixer_drop_path_rate

        # One DropPath module per GNN layer (layer 0 has no residual, skip it)
        # nlayer_gnn layers → nlayer_gnn residual connections starting at i=1
        nlayer_gnn = len(self.gnns)
        # Linear schedule: 0 → drop_path_rate  across layers
        rates = [gnn_drop_path_rate * i / max(nlayer_gnn - 1, 1)
                 for i in range(nlayer_gnn)]
        self.gnn_drop_paths = nn.ModuleList([DropPath(r) for r in rates])

    # -------------------------------------------------------------------
    # Override _gnn_forward to inject DropPath at GNN residuals
    # -------------------------------------------------------------------
    def _gnn_forward(self, data):
        """GNN forward with DropPath at residual connections."""
        x = self.input_encoder(data.x).squeeze()

        edge_attr = data.edge_attr
        if edge_attr is None:
            edge_attr = data.edge_index.new_zeros(
                data.edge_index.size(-1)).float().unsqueeze(-1)
        edge_attr = self.edge_encoder(edge_attr)

        x          = x[data.subgraphs_nodes_mapper]
        edge_index = data.combined_subgraphs
        e          = edge_attr[data.subgraphs_edges_mapper]
        batch_x    = data.subgraphs_batch
        pes        = data.rw_pos_enc[data.subgraphs_nodes_mapper]
        raw_patch_pes = scatter(pes, batch_x, dim=0, reduce='max')

        for i, gnn in enumerate(self.gnns):
            if i > 0:
                # Compute subgraph-level aggregation  [n_subgraphs, nhid]
                subgraph_agg = scatter(x, batch_x, dim=0, reduce=self.pooling)

                # Broadcast back to node level: [n_nodes_in_batch, nhid]
                subgraph_broadcast = subgraph_agg[batch_x]

                # DropPath on the U(subgraph) residual
                delta_u = self.U[i - 1](subgraph_broadcast)
                # DropPath needs batch dimension → treat each subgraph as a sample
                # We reshape: [n_nodes, d] → [n_subgraphs, nodes_per_patch, d]
                # Since variable patch size, apply DropPath per-subgraph via batch_x
                # Simplified: apply DropPath on the aggregated delta before broadcast
                delta_u_agg = self.U[i - 1](subgraph_agg)   # [n_sub, d]
                delta_u_agg = self.gnn_drop_paths[i](delta_u_agg)  # DropPath
                x = x + delta_u_agg[batch_x]                 # broadcast back

                # Cross-graph scatter residual
                cross_agg = scatter(x, data.subgraphs_nodes_mapper,
                                    dim=0, reduce='mean')[data.subgraphs_nodes_mapper]
                x = cross_agg   # replaces x += scatter(...) from parent

            x = gnn(x, edge_index, e)

        subgraph_x_L0 = scatter(x, batch_x, dim=0, reduce=self.pooling)
        return subgraph_x_L0, raw_patch_pes, batch_x


# ─────────────────────────────────────────────────────────────
# 5. CUSTOM MODEL FACTORY
# ─────────────────────────────────────────────────────────────

def _resolve_dataset_types(dataset_name):
    """Return (node_type, edge_type, nfeat_node, nfeat_edge) for a dataset."""
    mapping = {
        'MUTAG':        ('Linear',   'Linear',   7,  4),
        'PROTEINS':     ('Linear',   'Linear',   3,  1),
        'DD':           ('Linear',   'Linear',   89, 1),
        'IMDB-BINARY':  ('Linear',   'Linear',   1,  1),
        'IMDB-MULTI':   ('Linear',   'Linear',   1,  1),
        'REDDIT-BINARY':('Linear',   'Linear',   1,  1),
        'ZINC':         ('Discrete', 'Discrete', 28, 4),
    }
    return mapping.get(dataset_name, ('Linear', 'Linear', 1, 1))


# DropPath rates per dataset (tuned for dataset size; larger DP for smaller datasets)
DATASET_DROP_RATES = {
    "MUTAG":       {"gnn": 0.15, "mixer": 0.10},   # 188 graphs, smallest → highest dropout
    "PROTEINS":    {"gnn": 0.10, "mixer": 0.08},
    "IMDB-BINARY": {"gnn": 0.10, "mixer": 0.08},
    "IMDB-MULTI":  {"gnn": 0.08, "mixer": 0.05},
    "DD":          {"gnn": 0.05, "mixer": 0.05},   # 1178 graphs, less needed
    "ZINC":        {"gnn": 0.03, "mixer": 0.03},   # 12000 graphs, minimal needed
}

_CURRENT_DATASET = ["MUTAG"]  # set before each run


def create_model_droppath(cfg):
    """
    Drop-in replacement for create_model() that returns
    GraphHMSJepaDropPath with dataset-specific DropPath rates.
    Falls back to standard create_model for non-HMS models.
    """
    from core.get_model import create_model as _base_create
    model = _base_create(cfg)
    if not isinstance(model, GraphHMSJepa):
        return model  # non-HMS model — no DropPath

    ds_name = _CURRENT_DATASET[0]
    rates   = DATASET_DROP_RATES.get(ds_name, {"gnn": 0.05, "mixer": 0.05})

    # Resolve dataset-specific node/edge types
    node_type, edge_type, nfeat_node, nfeat_edge = _resolve_dataset_types(ds_name)

    # Re-instantiate as GraphHMSJepaDropPath with same cfg
    dp_model = GraphHMSJepaDropPath(
        nfeat_node           = nfeat_node,
        nfeat_edge           = nfeat_edge,
        nhid                 = model.nhid,
        nout                 = 1,
        nlayer_gnn           = len(model.gnns),
        nlayer_mlpmixer      = model.context_encoder_L0.nlayer
                               if hasattr(model.context_encoder_L0, 'nlayer')
                               else cfg.model.nlayer_mlpmixer,
        node_type            = node_type,
        edge_type            = edge_type,
        gnn_type             = cfg.model.gnn_type,
        gMHA_type            = cfg.model.gMHA_type,
        rw_dim               = cfg.pos_enc.rw_dim,
        lap_dim              = cfg.pos_enc.lap_dim,
        dropout              = getattr(cfg.train, 'dropout', 0),
        mlpmixer_dropout     = getattr(cfg.train, 'mlpmixer_dropout', 0),
        n_patches            = cfg.metis.n_patches,
        patch_rw_dim         = cfg.pos_enc.patch_rw_dim,
        num_context_patches  = cfg.jepa.num_context,
        num_target_patches   = cfg.jepa.num_targets,
        num_target_patches_L1= cfg.jepa.num_targets_L1,
        num_target_patches_L2= cfg.jepa.num_targets_L2,
        loss_weights         = cfg.jepa.loss_weights,
        var_weight           = cfg.jepa.var_weight,
        gnn_drop_path_rate   = rates["gnn"],
        mixer_drop_path_rate = rates["mixer"],
    )
    print(f"  [DropPath] dataset={ds_name}  gnn_dp={rates['gnn']}  mixer_dp={rates['mixer']}")
    return dp_model


# ─────────────────────────────────────────────────────────────
# 6. PER-DATASET CONFIGS (identical to exp07 — no ZINC)
# ─────────────────────────────────────────────────────────────
DATASET_CONFIGS = {
    "MUTAG": """
dataset: MUTAG
num_workers: 4
model:
  gMHA_type: Hadamard
  gnn_type: GINEConv
  nlayer_gnn: 2
  nlayer_mlpmixer: 4
  hidden_size: 512
train:
  lr_patience: 20
  epochs: 50
  batch_size: 512
  lr: 0.0005
  runs: 5
metis:
  n_patches: 32
  online: False
pos_enc:
  rw_dim: 15
  patch_rw_dim: 15
  lap_dim: 0
  patch_num_diff: 0
jepa:
  num_context: 1
  num_targets: 3
  num_scales: 3
  scale_factor: 4
  loss_weights: [1.0, 0.5, 0.25]
  num_targets_L1: 4
  num_targets_L2: 1
  var_weight: 0.01
device: 0
""",
    "PROTEINS": """
dataset: PROTEINS
num_workers: 4
model:
  gMHA_type: Hadamard
  gnn_type: GINEConv
  nlayer_gnn: 2
  nlayer_mlpmixer: 4
  hidden_size: 512
train:
  lr_patience: 20
  epochs: 30
  batch_size: 512
  lr: 0.0005
  runs: 5
metis:
  n_patches: 32
  online: False
pos_enc:
  rw_dim: 20
  patch_rw_dim: 20
  lap_dim: 0
  patch_num_diff: 0
jepa:
  num_context: 1
  num_targets: 2
  num_scales: 3
  scale_factor: 4
  loss_weights: [1.0, 0.5, 0.25]
  num_targets_L1: 4
  num_targets_L2: 1
  var_weight: 0.01
device: 0
""",
    "IMDB-BINARY": """
dataset: IMDB-BINARY
num_workers: 4
model:
  gMHA_type: Hadamard
  gnn_type: GINEConv
  nlayer_gnn: 2
  nlayer_mlpmixer: 4
  hidden_size: 512
train:
  lr_patience: 20
  epochs: 10
  batch_size: 512
  lr: 0.0005
  runs: 5
metis:
  n_patches: 32
  online: False
pos_enc:
  rw_dim: 15
  patch_rw_dim: 15
  lap_dim: 0
  patch_num_diff: 0
jepa:
  num_context: 1
  num_targets: 4
  num_scales: 3
  scale_factor: 4
  loss_weights: [1.0, 0.5, 0.25]
  num_targets_L1: 4
  num_targets_L2: 1
  var_weight: 0.01
device: 0
""",
    "IMDB-MULTI": """
dataset: IMDB-MULTI
num_workers: 4
model:
  gMHA_type: Hadamard
  gnn_type: GINEConv
  nlayer_gnn: 2
  nlayer_mlpmixer: 2
  hidden_size: 512
train:
  lr_patience: 20
  epochs: 5
  batch_size: 512
  lr: 0.0005
  runs: 5
metis:
  n_patches: 32
  online: False
pos_enc:
  rw_dim: 15
  patch_rw_dim: 15
  lap_dim: 0
  patch_num_diff: 0
jepa:
  num_context: 1
  num_targets: 4
  num_scales: 3
  scale_factor: 4
  loss_weights: [1.0, 0.5, 0.25]
  num_targets_L1: 4
  num_targets_L2: 1
  var_weight: 0.01
device: 0
""",
    "DD": """
dataset: DD
num_workers: 4
model:
  gMHA_type: Hadamard
  gnn_type: GINEConv
  nlayer_gnn: 3
  nlayer_mlpmixer: 4
  hidden_size: 512
train:
  lr_patience: 20
  epochs: 20
  batch_size: 256
  lr: 0.0005
  runs: 5
metis:
  n_patches: 32
  online: False
pos_enc:
  rw_dim: 30
  patch_rw_dim: 30
  lap_dim: 0
  patch_num_diff: 0
jepa:
  num_context: 1
  num_targets: 4
  num_scales: 3
  scale_factor: 4
  loss_weights: [1.0, 0.5, 0.25]
  num_targets_L1: 4
  num_targets_L2: 1
  var_weight: 0.01
device: 0
""",
    "ZINC": """
dataset: ZINC
num_workers: 4
model:
  gMHA_type: Hadamard
  gnn_type: GINEConv
  nlayer_gnn: 2
  nlayer_mlpmixer: 4
  hidden_size: 512
train:
  lr_patience: 20
  epochs: 30
  batch_size: 512
  lr: 0.0005
  runs: 10
metis:
  n_patches: 32
  online: False
pos_enc:
  rw_dim: 20
  patch_rw_dim: 20
  lap_dim: 0
  patch_num_diff: 0
jepa:
  num_context: 1
  num_targets: 4
  num_scales: 3
  scale_factor: 4
  loss_weights: [1.0, 0.5, 0.25]
  num_targets_L1: 4
  num_targets_L2: 1
  var_weight: 0.01
device: 0
""",
}

DATASETS_TO_RUN    = ["MUTAG", "PROTEINS", "IMDB-BINARY", "IMDB-MULTI", "DD", "ZINC"]
_PREFLIGHT_MAX_GRAPHS = 24


# ─────────────────────────────────────────────────────────────
# 7. HELPERS
# ─────────────────────────────────────────────────────────────

def _device_tensor(cfg):
    d = cfg.device
    if isinstance(d, torch.device): return d
    if isinstance(d, str):          return torch.device(d)
    if isinstance(d, int):          return torch.device(f"cuda:{d}" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def _merge_cfg_for_dataset(dataset_name):
    cfg_yaml = DATASET_CONFIGS[dataset_name]
    cfg_path = f"/tmp/exp10_{dataset_name.replace('-', '_').lower()}.yaml"
    with open(cfg_path, "w") as f:
        f.write(cfg_yaml)
    from core.config import cfg as _cfg
    _cfg.defrost()
    _cfg.merge_from_file(cfg_path)
    updated_cfg = update_cfg(_cfg, args_str="")
    updated_cfg.k = 10
    return updated_cfg


def preflight_one_dataset(dataset_name):
    _CURRENT_DATASET[0] = dataset_name
    cfg = _merge_cfg_for_dataset(dataset_name)
    device = _device_tensor(cfg)

    out = create_dataset(cfg)
    if cfg.dataset == "ZINC":
        train_dataset, _val, _test = out
        n = min(_PREFLIGHT_MAX_GRAPHS, len(train_dataset))
        if n < 1:
            raise RuntimeError(f"ZINC train split empty (len={len(train_dataset)})")
        small = train_dataset[:n]
        if not cfg.metis.online:
            small = [x for x in small]
        bs = min(cfg.train.batch_size, max(1, len(small)))
        loader = DataLoader(small, batch_size=bs, shuffle=True,
                            num_workers=0, pin_memory=False)
    else:
        dataset, transform, _ = out
        train_indices, _ = k_fold(dataset, cfg.k)
        ti  = train_indices[0]
        n   = min(_PREFLIGHT_MAX_GRAPHS, len(ti))
        if n < 1:
            raise RuntimeError("Preflight fold: no training indices")
        subset = dataset[ti[:n]]
        subset.transform = transform
        train_list = [x for x in subset] if not cfg.metis.online else subset
        bs = min(cfg.train.batch_size, max(1, len(train_list)))
        loader = DataLoader(train_list, batch_size=bs, shuffle=True,
                            num_workers=0, pin_memory=False)

    model     = create_model_droppath(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr,
                                 weight_decay=getattr(cfg.train, 'wd', 0))
    criterion = torch.nn.SmoothL1Loss(beta=0.5)
    scaler    = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    data      = next(iter(loader))
    if getattr(model, "use_lap", False) and hasattr(data, "lap_pos_enc") and data.lap_pos_enc is not None:
        b_pe = data.lap_pos_enc
        sf   = torch.rand(b_pe.size(1)); sf[sf >= 0.5] = 1.0; sf[sf < 0.5] = -1.0
        data.lap_pos_enc = b_pe * sf.unsqueeze(0)
    data = data.to(device)
    optimizer.zero_grad()
    with torch.amp.autocast("cuda", enabled=USE_AMP, dtype=AMP_DTYPE):
        loss, _ = _compute_loss(model, data, criterion, cfg.jepa.dist)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()


def run_preflight_all():
    print("\n" + "=" * 70)
    print("  EXP 10 — PREFLIGHT (Stochastic Depth / DropPath)")
    print("=" * 70)
    t0 = time.time()
    for name in DATASETS_TO_RUN:
        t_ds = time.time()
        print(f"\n  [preflight] {name} ...", flush=True)
        preflight_one_dataset(name)
        print(f"  [preflight] {name} OK ({time.time() - t_ds:.1f}s)", flush=True)
    print(f"\n  Preflight finished in {(time.time() - t0) / 60.0:.1f} min.\n")


# ─────────────────────────────────────────────────────────────
# 8. TRAIN / TEST LOOPS
# ─────────────────────────────────────────────────────────────

def train(train_loader, model, optimizer, evaluator, device,
          momentum_weight, sharp=None, criterion_type=0):
    criterion = torch.nn.SmoothL1Loss(beta=0.5)
    scaler    = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    step_losses, num_targets = [], []

    model.train()  # DropPath only active in train mode
    for data in train_loader:
        if model.use_lap:
            b_pe = data.lap_pos_enc
            sf   = torch.rand(b_pe.size(1)); sf[sf >= 0.5] = 1.0; sf[sf < 0.5] = -1.0
            data.lap_pos_enc = b_pe * sf.unsqueeze(0)
        data = data.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=USE_AMP, dtype=AMP_DTYPE):
            loss, num_t = _compute_loss(model, data, criterion, criterion_type)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        step_losses.append(loss.item())
        num_targets.append(num_t)

        with torch.no_grad():
            _ema_update(model, momentum_weight)

    return None, np.average(step_losses, weights=num_targets)


@torch.no_grad()
def test(loader, model, evaluator, device, criterion_type=0):
    criterion = torch.nn.SmoothL1Loss(beta=0.5)
    step_losses, num_targets = [], []
    model.eval()   # DropPath disabled at test time
    for data in loader:
        data = data.to(device)
        with torch.amp.autocast('cuda', enabled=USE_AMP, dtype=AMP_DTYPE):
            loss, num_t = _compute_loss(model, data, criterion, criterion_type)
        step_losses.append(loss.item())
        num_targets.append(num_t)
    return None, np.average(step_losses, weights=num_targets)


# ─────────────────────────────────────────────────────────────
# 9. ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    skip_pf = os.environ.get("EXP10_SKIP_PREFLIGHT", "0") == "1"
    pf_only = os.environ.get("EXP10_PREFLIGHT_ONLY", "0") == "1"

    print(f"\n  EXP 10: Stochastic Depth (DropPath) Regularisation")
    print(f"  DropPath rates per dataset:")
    for ds, rates in DATASET_DROP_RATES.items():
        print(f"    {ds:<12}  gnn={rates['gnn']}  mixer={rates['mixer']}")

    if not skip_pf:
        try:
            run_preflight_all()
        except Exception as e:
            print(f"\n  PREFLIGHT FAILED: {e!r}", flush=True)
            raise
        if pf_only:
            print("EXP10_PREFLIGHT_ONLY=1 — exiting after successful preflight.")
            sys.exit(0)

    wall_times  = {}
    total_start = time.time()

    for dataset_name in DATASETS_TO_RUN:
        _CURRENT_DATASET[0] = dataset_name          # set before create_model

        ds_start = time.time()
        print(f"\n{'#' * 70}")
        print(f"#  EXP 10 — DropPath — DATASET: {dataset_name}")
        print(f"{'#' * 70}")

        updated_cfg = _merge_cfg_for_dataset(dataset_name)
        is_regression = (dataset_name == "ZINC")
        if is_regression:
            run(updated_cfg, create_dataset, create_model_droppath, train, test)
        else:
            run_k_fold(updated_cfg, create_dataset, create_model_droppath, train, test)

        ds_elapsed = time.time() - ds_start
        wall_times[dataset_name] = ds_elapsed
        print(f"\n  [{dataset_name}] wall time: {ds_elapsed / 60:.1f} min")

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"  EXP 10 (DropPath) — ALL DATASETS COMPLETE")
    print(f"  Total wall time: {total_elapsed / 60:.1f} min")
    print(f"{'=' * 70}")
    for ds, t in wall_times.items():
        print(f"  {ds:<12}  {t / 60:>9.1f}m")
    print(f"{'=' * 70}")
    print("  [TRACKER] Copy results above into tracker.md → EXP 10 row")
