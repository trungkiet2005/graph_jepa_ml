# ============================================================
# EXP 11 — HMS-JEPA + Poincaré Ball Prediction Space
# ============================================================
# Novelty: Replace the Lorentzian (cosh/sinh 2D unit hyperbola)
# prediction target with FULL-DIMENSIONAL Poincaré Ball embeddings.
# The predictor outputs an nhid-dimensional vector then maps it to
# the Poincaré ball via the exponential map. The target encoder also
# maps to the ball. Loss = Poincaré geodesic distance.
#
# Key insight: The paper uses a 2D Lorentzian bottleneck → catastrophic
# dimensionality reduction (512D → 2D). Poincaré ball in full nhid
# dimension preserves all structural information while still benefiting
# from hyperbolic geometry for hierarchical graph representations.
#
# All model code is SELF-CONTAINED in this file (no repo modification).
#
# Theoretical grounding:
#   - Nickel & Kiela, "Poincaré Embeddings for Learning Hierarchical
#     Representations", NeurIPS 2017.
#   - Chami et al., "Hyperbolic Graph Convolutional Neural Networks",
#     NeurIPS 2019.
#   - Chen et al., "Fully Hyperbolic Neural Networks", ACL 2022.
#   - Gao et al., "Curvature Generation in Curved Spaces for Few-Shot
#     Learning", ICCV 2021.
#   - Skenderi et al., Graph-JEPA (TMLR 2025) — ablation in Table 3
#     shows Hyperbolic > Euclidean; Poincaré scores 89.43 vs 91.25.
#     We target exceeding the Lorentzian (91.25) with full-dim Poincaré.
#
# Datasets: MUTAG, PROTEINS, IMDB-BINARY, IMDB-MULTI, DD, ZINC
# ============================================================

import os, sys, subprocess, time, math

REPO_URL = "https://github.com/trungkiet2005/graph_jepa_ml.git"
REPO_DIR = "/kaggle/working/graph_jepa_ml"
if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
else:
    subprocess.run(["git", "-C", REPO_DIR, "pull", "--ff-only"], check=True)
os.chdir(REPO_DIR); sys.path.insert(0, REPO_DIR)
subprocess.run(["apt-get", "install", "-y", "libmetis-dev"], check=True, capture_output=True)
os.environ["METIS_DLL"] = "/usr/lib/x86_64-linux-gnu/libmetis.so"

import torch
torch_ver = torch.__version__
subprocess.run(["pip", "install", "-q", "torch-scatter", "torch-sparse", "torch-cluster",
    "torch-geometric", "-f", f"https://data.pyg.org/whl/torch-{torch_ver}.html"], check=True)
subprocess.run(["pip", "install", "-q", "yacs", "tensorboard", "networkx", "einops", "metis", "ogb"], check=True)

# ─────────────────────────────────────────────────────────────
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch_scatter import scatter
from core.config import cfg, update_cfg
from core.get_data import create_dataset
from core.get_model import create_model
from torch_geometric.loader import DataLoader
from core.trainer import run_k_fold, run, k_fold
from core.tracker_footer import print_exp_tracker_footer
from core.model import GraphHMSJepa
from core.model_utils.elements import MLP
from core.model_utils.feature_encoder import FeatureEncoder
from core.model_utils.gnn import GNN
import core.model_utils.gMHA_wrapper as gMHA_wrapper
from train.zinc import _ema_update

torch.backends.cudnn.benchmark = True
USE_AMP = torch.cuda.is_available()
AMP_DTYPE = torch.bfloat16

# ─────────────────────────────────────────────────────────────
# POINCARÉ BALL GEOMETRY  (c = curvature, default c=1)
# ─────────────────────────────────────────────────────────────
POINCARE_C = 1.0    # curvature of the Poincaré ball
POINCARE_EPS = 1e-5  # numerical stability clamp

def poincare_clamp(x, c=POINCARE_C):
    """Project x to the open Poincaré ball: ||x|| < 1/sqrt(c) - eps."""
    max_norm = (1.0 / math.sqrt(c)) - POINCARE_EPS
    norm = x.norm(dim=-1, keepdim=True).clamp(min=POINCARE_EPS)
    return x * (max_norm / norm.clamp(min=max_norm))

def expmap0(v, c=POINCARE_C):
    """Exponential map at origin: maps tangent vector v → Poincaré ball."""
    norm_v = v.norm(dim=-1, keepdim=True).clamp(min=POINCARE_EPS)
    sqrt_c = math.sqrt(c)
    tanh_arg = sqrt_c * norm_v
    # Clamp to avoid tanh saturation
    tanh_arg = tanh_arg.clamp(max=15.0)
    return (torch.tanh(tanh_arg) / (sqrt_c * norm_v)) * v

def logmap0(y, c=POINCARE_C):
    """Logarithmic map at origin: maps Poincaré ball point → tangent space."""
    norm_y = y.norm(dim=-1, keepdim=True).clamp(min=POINCARE_EPS)
    sqrt_c = math.sqrt(c)
    atanh_arg = sqrt_c * norm_y
    atanh_arg = atanh_arg.clamp(max=1.0 - POINCARE_EPS)
    return (torch.atanh(atanh_arg) / (sqrt_c * norm_y)) * y

def poincare_distance(x, y, c=POINCARE_C):
    """
    Geodesic distance on the Poincaré ball between x and y.
    d(x,y) = (2/sqrt(c)) * atanh(sqrt(c) * ||(-x) ⊕ y||)
    where ⊕ is Möbius addition.
    """
    # Möbius addition: (-x) ⊕ y
    sqrt_c = math.sqrt(c)
    x_norm_sq = (x * x).sum(dim=-1, keepdim=True).clamp(min=0)
    y_norm_sq = (y * y).sum(dim=-1, keepdim=True).clamp(min=0)
    xy_dot    = (x * y).sum(dim=-1, keepdim=True)

    num   = (1 + 2*c*xy_dot + c*y_norm_sq) * (-x) + (1 - c*x_norm_sq) * y
    denom = 1 + 2*c*(-xy_dot) + c**2 * x_norm_sq * y_norm_sq
    mobius = num / denom.clamp(min=POINCARE_EPS)

    mobius_norm = mobius.norm(dim=-1).clamp(min=0, max=1.0/sqrt_c - POINCARE_EPS)
    atanh_arg   = sqrt_c * mobius_norm
    dist = (2.0 / sqrt_c) * torch.atanh(atanh_arg.clamp(max=1.0 - POINCARE_EPS))
    return dist  # [B, nT]

# ─────────────────────────────────────────────────────────────
# POINCARÉ HMS-JEPA MODEL
# ─────────────────────────────────────────────────────────────
class GraphHMSJepaPoincare(GraphHMSJepa):
    """
    HMS-JEPA with full-dimensional Poincaré ball prediction space.
    Overrides the forward() to:
      (a) Map target encoder output to Poincaré ball via expmap0
      (b) Map predictor output to Poincaré ball via expmap0
      (c) Return Poincaré-space representations for geodesic-distance loss
    Also adds a trainable curvature parameter (learnable c) per scale.
    """
    def __init__(self, *args, learn_curvature=True, **kwargs):
        super().__init__(*args, **kwargs)
        # Learnable curvature per scale (initialized to c=1)
        if learn_curvature:
            self.log_c_L0 = nn.Parameter(torch.zeros(1))  # c = exp(log_c)
            self.log_c_L1 = nn.Parameter(torch.zeros(1))
            self.log_c_L2 = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer('log_c_L0', torch.zeros(1))
            self.register_buffer('log_c_L1', torch.zeros(1))
            self.register_buffer('log_c_L2', torch.zeros(1))

        # Poincaré output gate: learned scale before expmap
        self.poincare_gate_L0 = nn.Linear(self.nhid, self.nhid, bias=False)
        self.poincare_gate_L1 = nn.Linear(self.nhid, self.nhid, bias=False)
        self.poincare_gate_L2 = nn.Linear(self.nhid, self.nhid, bias=False)
        nn.init.eye_(self.poincare_gate_L0.weight)
        nn.init.eye_(self.poincare_gate_L1.weight)
        nn.init.eye_(self.poincare_gate_L2.weight)

    def _to_poincare(self, x, log_c, gate):
        """Project tensor x to Poincaré ball using learned curvature and gate."""
        c = torch.exp(log_c).clamp(min=0.01, max=10.0).item()
        v = gate(x)                     # tangent vector
        p = expmap0(v, c=c)             # → Poincaré ball
        p = poincare_clamp(p, c=c)      # safety clamp
        return p, c

    def forward(self, data):
        """
        Forward that returns Poincaré-space (target, prediction) pairs.
        We reuse the parent's forward to get (tgt, pred) Euclidean tensors,
        then project both to the Poincaré ball.
        """
        (tx0, ty0), (tx1, ty1), (tx2, ty2) = super().forward(data)

        # Project targets to Poincaré ball (target encoder runs in no_grad)
        # tx is already detached from target_encoder (due to no_grad in parent)
        p_tx0, c0 = self._to_poincare(tx0, self.log_c_L0, self.poincare_gate_L0)
        p_tx1, c1 = self._to_poincare(tx1, self.log_c_L1, self.poincare_gate_L1)
        p_tx2, c2 = self._to_poincare(tx2, self.log_c_L2, self.poincare_gate_L2)

        # Project predictions to Poincaré ball
        p_ty0, _  = self._to_poincare(ty0, self.log_c_L0, self.poincare_gate_L0)
        p_ty1, _  = self._to_poincare(ty1, self.log_c_L1, self.poincare_gate_L1)
        p_ty2, _  = self._to_poincare(ty2, self.log_c_L2, self.poincare_gate_L2)

        # Store curvatures for logging
        self._last_c = (c0, c1, c2)
        return (p_tx0, p_ty0), (p_tx1, p_ty1), (p_tx2, p_ty2)


# ─────────────────────────────────────────────────────────────
# CUSTOM LOSS: Poincaré Geodesic Distance
# ─────────────────────────────────────────────────────────────
def _poincare_jepa_loss(model, data):
    """
    Forward pass + Poincaré geodesic distance loss across 3 HMS scales.
    Returns (loss, num_targets).
    """
    if not isinstance(model, GraphHMSJepaPoincare):
        # fallback to standard SmoothL1 for non-Poincaré models
        from train.zinc import _compute_loss
        return _compute_loss(model, data, nn.SmoothL1Loss(beta=0.5), 0)

    (tx0, ty0), (tx1, ty1), (tx2, ty2) = model(data)
    # Poincaré distances: shape [B*nT]
    d0 = poincare_distance(ty0, tx0.detach())    # prediction vs target
    d1 = poincare_distance(ty1, tx1.detach())
    d2 = poincare_distance(ty2, tx2.detach())

    w = model.loss_weights
    loss = w[0] * d0.mean() + w[1] * d1.mean() + w[2] * d2.mean()

    # Variance regularisation in Poincaré space (norm diversity)
    tx0_norms = tx0.detach().norm(dim=-1)         # [B*nT]
    var_reg = torch.relu(0.5 - tx0_norms.std())   # encourage spread
    loss = loss + model.var_weight * var_reg

    num_t = len(ty0)
    return loss, num_t


# ─────────────────────────────────────────────────────────────
# MODEL FACTORY: always return GraphHMSJepaPoincare
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


def create_model_poincare(cfg):
    base = create_model(cfg)
    if not isinstance(base, GraphHMSJepa):
        return base
    nlayer_gnn      = len(base.gnns)
    nlayer_mlpmixer = cfg.model.nlayer_mlpmixer
    node_type, edge_type, nfeat_node, nfeat_edge = _resolve_dataset_types(cfg.dataset)
    model = GraphHMSJepaPoincare(
        nfeat_node            = nfeat_node,
        nfeat_edge            = nfeat_edge,
        nhid                  = base.nhid,
        nout                  = 1,
        nlayer_gnn            = nlayer_gnn,
        nlayer_mlpmixer       = nlayer_mlpmixer,
        node_type             = node_type,
        edge_type             = edge_type,
        gnn_type              = cfg.model.gnn_type,
        gMHA_type             = cfg.model.gMHA_type,
        rw_dim                = cfg.pos_enc.rw_dim,
        lap_dim               = cfg.pos_enc.lap_dim,
        dropout               = getattr(cfg.train, 'dropout', 0),
        mlpmixer_dropout      = getattr(cfg.train, 'mlpmixer_dropout', 0),
        n_patches             = cfg.metis.n_patches,
        patch_rw_dim          = cfg.pos_enc.patch_rw_dim,
        num_context_patches   = cfg.jepa.num_context,
        num_target_patches    = cfg.jepa.num_targets,
        num_target_patches_L1 = cfg.jepa.num_targets_L1,
        num_target_patches_L2 = cfg.jepa.num_targets_L2,
        loss_weights          = cfg.jepa.loss_weights,
        var_weight            = cfg.jepa.var_weight,
        learn_curvature       = True,
    )
    return model


# ─────────────────────────────────────────────────────────────
# DATASET CONFIGS (no ZINC)
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

DATASETS_TO_RUN = ["MUTAG", "PROTEINS", "IMDB-BINARY", "IMDB-MULTI", "DD", "ZINC"]
_PREFLIGHT_MAX_GRAPHS = 24


def _device_tensor(cfg):
    d = cfg.device
    if isinstance(d, torch.device): return d
    if isinstance(d, str): return torch.device(d)
    if isinstance(d, int): return torch.device(f"cuda:{d}" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def _merge_cfg_for_dataset(dataset_name):
    cfg_yaml = DATASET_CONFIGS[dataset_name]
    cfg_path = f"/tmp/exp11_{dataset_name.replace('-','_').lower()}.yaml"
    with open(cfg_path, "w") as f: f.write(cfg_yaml)
    from core.config import cfg as _cfg
    _cfg.defrost(); _cfg.merge_from_file(cfg_path)
    updated_cfg = update_cfg(_cfg, args_str=""); updated_cfg.k = 10
    return updated_cfg


def preflight_one_dataset(dataset_name):
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
        loader = DataLoader(small, batch_size=bs, shuffle=True, num_workers=0, pin_memory=False)
    else:
        dataset, transform, _ = out
        train_indices, _ = k_fold(dataset, cfg.k)
        ti = train_indices[0]
        n  = min(_PREFLIGHT_MAX_GRAPHS, len(ti))
        subset = dataset[ti[:n]]; subset.transform = transform
        train_list = [x for x in subset]
        bs = min(cfg.train.batch_size, max(1, len(train_list)))
        loader = DataLoader(train_list, batch_size=bs, shuffle=True, num_workers=0, pin_memory=False)
    model = create_model_poincare(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    data = next(iter(loader)).to(device)
    optimizer.zero_grad()
    with torch.amp.autocast("cuda", enabled=USE_AMP, dtype=AMP_DTYPE):
        loss, _ = _poincare_jepa_loss(model, data)
    scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()


def run_preflight_all():
    print("\n" + "="*70)
    print("  EXP 11 — PREFLIGHT (Poincaré Ball, full-dim, learnable curvature)")
    print("="*70)
    t0 = time.time()
    for name in DATASETS_TO_RUN:
        t_ds = time.time()
        print(f"\n  [preflight] {name} ...", flush=True)
        preflight_one_dataset(name)
        print(f"  [preflight] {name} OK ({time.time()-t_ds:.1f}s)", flush=True)
    print(f"\n  Preflight finished in {(time.time()-t0)/60:.1f} min.\n")


# ─────────────────────────────────────────────────────────────
# TRAIN / TEST
# ─────────────────────────────────────────────────────────────
def train(train_loader, model, optimizer, evaluator, device,
          momentum_weight, sharp=None, criterion_type=0):
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    step_losses, num_targets = [], []
    for data in train_loader:
        if model.use_lap:
            b_pe = data.lap_pos_enc
            sf = torch.rand(b_pe.size(1)); sf[sf >= 0.5] = 1.0; sf[sf < 0.5] = -1.0
            data.lap_pos_enc = b_pe * sf.unsqueeze(0)
        data = data.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=USE_AMP, dtype=AMP_DTYPE):
            loss, num_t = _poincare_jepa_loss(model, data)
        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update()
        step_losses.append(loss.item()); num_targets.append(num_t)
        with torch.no_grad(): _ema_update(model, momentum_weight)
    return None, np.average(step_losses, weights=num_targets)


@torch.no_grad()
def test(loader, model, evaluator, device, criterion_type=0):
    step_losses, num_targets = [], []
    for data in loader:
        data = data.to(device)
        with torch.amp.autocast('cuda', enabled=USE_AMP, dtype=AMP_DTYPE):
            loss, num_t = _poincare_jepa_loss(model, data)
        step_losses.append(loss.item()); num_targets.append(num_t)
    return None, np.average(step_losses, weights=num_targets)


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    skip_pf = os.environ.get("EXP11_SKIP_PREFLIGHT", "0") == "1"
    pf_only = os.environ.get("EXP11_PREFLIGHT_ONLY", "0") == "1"
    print(f"\n  EXP 11: Full-dim Poincaré Ball JEPA | learnable curvature c per scale")

    if not skip_pf:
        try: run_preflight_all()
        except Exception as e: print(f"\n  PREFLIGHT FAILED: {e!r}"); raise
        if pf_only: print("EXP11_PREFLIGHT_ONLY=1"); sys.exit(0)

    wall_times = {}; tracker_results = {}; total_start = time.time()
    for dataset_name in DATASETS_TO_RUN:
        ds_start = time.time()
        print(f"\n{'#'*70}\n#  EXP 11 — Poincaré — DATASET: {dataset_name}\n{'#'*70}")
        updated_cfg = _merge_cfg_for_dataset(dataset_name)
        is_regression = (dataset_name == "ZINC")
        if is_regression:
            tracker_results[dataset_name] = run(updated_cfg, create_dataset, create_model_poincare, train, test)
        else:
            tracker_results[dataset_name] = run_k_fold(updated_cfg, create_dataset, create_model_poincare, train, test)
        ds_elapsed = time.time() - ds_start
        wall_times[dataset_name] = ds_elapsed
        print(f"\n  [{dataset_name}] wall time: {ds_elapsed/60:.1f} min")

    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}\n  EXP 11 (Poincaré) COMPLETE | Total: {total_elapsed/60:.1f} min\n{'='*70}")
    for ds, t in wall_times.items(): print(f"  {ds:<12}  {t/60:>9.1f}m")
    print_exp_tracker_footer(11, "Full-dim Poincaré Ball JEPA", tracker_results)
