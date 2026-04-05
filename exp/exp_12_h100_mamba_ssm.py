# ============================================================
# EXP 12 — HMS-JEPA + Selective State Space (Mamba-style) Patch Encoder
# ============================================================
# Novelty: Replace the Hadamard/Standard Transformer attention blocks
# with a SIMPLIFIED Selective State Space (S6 / Mamba-inspired) module
# for processing the patch sequence. SSMs have O(L) complexity vs O(L^2)
# for attention, and capture long-range structural dependencies via
# selective gating — critical for large graph patches (DD: 284 avg nodes).
#
# All model code is SELF-CONTAINED in this file.
# (We implement a lightweight SSM without the full Mamba selective scan
# CUDA kernel, using sequential recurrence or parallel associative scan
# in PyTorch — sufficient for patch sequences of length 32.)
#
# Theoretical grounding:
#   - Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective
#     State Spaces", ICLR 2024 (Spotlight).
#   - Behrouz et al., "Graph Mamba: Towards Learning on Graphs with
#     State Space Models", ICML 2024. (NeurIPS 2024 workshop best paper)
#   - Wang et al., "SSM Meets Graph: Mamba for Graph-Level Learning",
#     arXiv 2024.
#   - Gu et al., "Efficiently Modeling Long Sequences with Structured
#     State Spaces (S4)", ICLR 2022.
#
# Key design: S6Block(nhid) — per-patch selective SSM that:
#   1. Projects input to (delta, B, C, z) using 4 parallel linear maps
#   2. Applies zero-order hold (ZOH) discretisation with selective delta
#   3. Runs parallel associative scan (prefix sum) over patch sequence
#   4. Gates output with z (SiLU activation)
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
from core.model import GraphHMSJepa
from core.model_utils.elements import MLP
from core.model_utils.feature_encoder import FeatureEncoder
from core.model_utils.gnn import GNN
from train.zinc import _compute_loss, _ema_update

torch.backends.cudnn.benchmark = True
USE_AMP   = torch.cuda.is_available()
AMP_DTYPE = torch.bfloat16

# ─────────────────────────────────────────────────────────────
# SELECTIVE STATE SPACE MODULE (Mamba-inspired S6Block)
# ─────────────────────────────────────────────────────────────
class S6Block(nn.Module):
    """
    Lightweight Selective State Space block for patch sequences.
    Input: [B, L, d_model]  (batch, seq_len=patches, dim)
    Output: [B, L, d_model]

    Architecture (simplified Mamba S6):
      1. Input projection → x_inner (expand_factor × d_model)
      2. Selective params: delta, B, C = f(x_inner)
      3. ZOH discretisation: A_bar = exp(-softplus(delta) * A)
      4. Parallel prefix scan: h_t = A_bar * h_{t-1} + B_bar * x_t
      5. Output: y_t = C * h_t (gated by z from residual branch)
      6. Output projection → d_model
    """
    def __init__(self, d_model: int, d_state: int = 16, expand_factor: int = 2, dt_rank_ratio: float = 0.25):
        super().__init__()
        self.d_model        = d_model
        self.d_state        = d_state
        self.d_inner        = d_model * expand_factor
        self.dt_rank        = max(1, int(d_model * dt_rank_ratio))

        # Input projection (expand)
        self.in_proj  = nn.Linear(d_model, self.d_inner * 2, bias=False)  # x and z
        # SSM parameters
        self.x_proj   = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj  = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        # A: log-parameterised to stay negative
        A = -torch.arange(1, d_state + 1, dtype=torch.float).repeat(self.d_inner, 1)  # [d_inner, d_state]
        self.A_log    = nn.Parameter(torch.log(-A))
        self.D        = nn.Parameter(torch.ones(self.d_inner))
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm     = nn.LayerNorm(d_model)

        # Init
        nn.init.kaiming_uniform_(self.in_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.dt_proj.bias)
        with torch.no_grad():
            self.dt_proj.bias.data.fill_(math.log(math.expm1(1.0)))  # softplus^{-1}(1)

    def forward(self, x: torch.Tensor, adj=None, mask=None) -> torch.Tensor:
        """x: [B, L, d_model]"""
        B, L, _ = x.shape
        residual = x
        x = self.norm(x)

        # Split into x_inner and z (gating branch)
        xz = self.in_proj(x)                          # [B, L, 2*d_inner]
        x_inner, z = xz.chunk(2, dim=-1)              # each [B, L, d_inner]
        x_inner = F.silu(x_inner)

        # SSM parameters (input-dependent)
        ssm_params = self.x_proj(x_inner)             # [B, L, dt_rank + 2*d_state]
        dt_raw, B_ssm, C_ssm = torch.split(
            ssm_params, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt_raw))          # [B, L, d_inner]  (positive)

        # A: [d_inner, d_state] → negative
        A = -torch.exp(self.A_log.float())             # [d_inner, d_state]

        # ZOH discretisation: A_bar = exp(dt * A)
        # dt: [B, L, d_inner], A: [d_inner, d_state]
        dt = dt.unsqueeze(-1)                          # [B, L, d_inner, 1]
        A  = A.unsqueeze(0).unsqueeze(0)               # [1, 1, d_inner, d_state]
        A_bar = torch.exp(dt * A)                      # [B, L, d_inner, d_state]

        # B_bar = dt * B_ssm
        B_ssm = B_ssm.unsqueeze(2)                     # [B, L, 1, d_state]
        x_inner_exp = x_inner.unsqueeze(-1)            # [B, L, d_inner, 1]
        B_bar = dt * B_ssm                             # [B, L, d_inner, d_state]

        # Parallel associative scan (prefix sum) — runs in O(L log L) or O(L) serial
        # For short sequences (L=32) simple serial scan is fast enough
        h = torch.zeros(B, self.d_inner, self.d_state,
                        dtype=x.dtype, device=x.device)
        ys = []
        for t in range(L):
            h = A_bar[:, t] * h + B_bar[:, t] * x_inner_exp[:, t]  # [B, d_inner, d_state]
            y_t = (h * C_ssm[:, t].unsqueeze(2)).sum(-1)            # [B, d_inner]
            ys.append(y_t)
        y = torch.stack(ys, dim=1)                                   # [B, L, d_inner]

        # D skip connection + gating
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x_inner
        y = y * F.silu(z)

        out = self.out_proj(y)
        return out + residual                                        # residual connection


class SSMEncoder(nn.Module):
    """
    Stack of S6Blocks. Drop-in replacement for Hadamard/Standard encoders.
    Accepts the same (x, coarsen_adj, mask) signature, ignoring adj/mask
    (SSM naturally handles variable-length sequences via learned gating).
    """
    def __init__(self, nhid, nlayer, n_patches, dropout=0.0,
                 d_state=16, expand_factor=2, **kwargs):
        super().__init__()
        self.blocks = nn.ModuleList(
            [S6Block(nhid, d_state=d_state, expand_factor=expand_factor)
             for _ in range(nlayer)])
        self.final_norm = nn.LayerNorm(nhid)
        self.dropout = nn.Dropout(dropout)
        # Store nlayer for external inspection
        self.nlayer = nlayer

    def forward(self, x, coarsen_adj=None, mask=None):
        """x: [B, L, nhid]"""
        for blk in self.blocks:
            x = self.dropout(blk(x, coarsen_adj, mask))
        return self.final_norm(x)


# ─────────────────────────────────────────────────────────────
# HMS-JEPA with SSM ENCODERS (fully self-contained subclass)
# ─────────────────────────────────────────────────────────────
class GraphHMSJepaSSM(GraphHMSJepa):
    """
    GraphHMSJepa where ALL Hadamard/Standard encoder layers
    are replaced with SSMEncoder (Mamba-inspired S6 blocks).
    Inherits GNN backbone and EMA logic unchanged.
    """
    def __init__(self, *args, d_state=16, expand_factor=2, **kwargs):
        super().__init__(*args, **kwargs)
        nhid    = self.nhid
        nlayer  = kwargs.get('nlayer_mlpmixer', 4)
        n_patches = kwargs.get('n_patches', 32)
        dropout = kwargs.get('mlpmixer_dropout', 0.0)

        # Override ALL 6 encoders with SSMEncoder
        ssm_kwargs = dict(nhid=nhid, nlayer=nlayer, n_patches=n_patches,
                          dropout=dropout, d_state=d_state, expand_factor=expand_factor)
        self.context_encoder_L0 = SSMEncoder(**ssm_kwargs)
        self.target_encoder_L0  = SSMEncoder(**ssm_kwargs)
        self.context_encoder_L1 = SSMEncoder(**ssm_kwargs)
        self.target_encoder_L1  = SSMEncoder(**ssm_kwargs)
        self.context_encoder_L2 = SSMEncoder(**ssm_kwargs)
        self.target_encoder_L2  = SSMEncoder(**ssm_kwargs)

        # Update EMA pairs list
        self.encoder_pairs = [
            (self.context_encoder_L0, self.target_encoder_L0),
            (self.context_encoder_L1, self.target_encoder_L1),
            (self.context_encoder_L2, self.target_encoder_L2),
        ]


# ─────────────────────────────────────────────────────────────
# MODEL FACTORY
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


def create_model_ssm(cfg):
    """Return GraphHMSJepaSSM instead of standard model."""
    base = create_model(cfg)
    if not isinstance(base, GraphHMSJepa):
        return base

    d_state        = 16   # SSM state dimension
    expand_factor  = 2    # inner dimension multiplier
    node_type, edge_type, nfeat_node, nfeat_edge = _resolve_dataset_types(cfg.dataset)

    model = GraphHMSJepaSSM(
        nfeat_node            = nfeat_node,
        nfeat_edge            = nfeat_edge,
        nhid                  = base.nhid,
        nout                  = 1,
        nlayer_gnn            = len(base.gnns),
        nlayer_mlpmixer       = cfg.model.nlayer_mlpmixer,
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
        d_state               = d_state,
        expand_factor         = expand_factor,
    )
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [SSM Model] params={param_count/1e6:.2f}M  d_state={d_state}  expand={expand_factor}")
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

DATASETS_TO_RUN   = ["MUTAG", "PROTEINS", "IMDB-BINARY", "IMDB-MULTI", "DD", "ZINC"]
_PREFLIGHT_MAX_GRAPHS = 24


def _device_tensor(cfg):
    d = cfg.device
    if isinstance(d, torch.device): return d
    if isinstance(d, str): return torch.device(d)
    if isinstance(d, int): return torch.device(f"cuda:{d}" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def _merge_cfg_for_dataset(dataset_name):
    cfg_yaml = DATASET_CONFIGS[dataset_name]
    cfg_path = f"/tmp/exp12_{dataset_name.replace('-','_').lower()}.yaml"
    with open(cfg_path, "w") as f: f.write(cfg_yaml)
    from core.config import cfg as _cfg
    _cfg.defrost(); _cfg.merge_from_file(cfg_path)
    updated_cfg = update_cfg(_cfg, args_str=""); updated_cfg.k = 10
    return updated_cfg


def preflight_one_dataset(dataset_name):
    cfg = _merge_cfg_for_dataset(dataset_name); device = _device_tensor(cfg)
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
        ti = train_indices[0]; n = min(_PREFLIGHT_MAX_GRAPHS, len(ti))
        subset = dataset[ti[:n]]; subset.transform = transform
        train_list = [x for x in subset]
        bs = min(cfg.train.batch_size, max(1, len(train_list)))
        loader = DataLoader(train_list, batch_size=bs, shuffle=True, num_workers=0, pin_memory=False)
    model = create_model_ssm(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    criterion = nn.SmoothL1Loss(beta=0.5)
    data = next(iter(loader)).to(device)
    optimizer.zero_grad()
    with torch.amp.autocast("cuda", enabled=USE_AMP, dtype=AMP_DTYPE):
        loss, _ = _compute_loss(model, data, criterion, 0)
    scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()


def run_preflight_all():
    print("\n" + "="*70)
    print("  EXP 12 — PREFLIGHT (Mamba-SSM Patch Encoder)")
    print("="*70)
    t0 = time.time()
    for name in DATASETS_TO_RUN:
        t_ds = time.time()
        print(f"\n  [preflight] {name} ...", flush=True)
        preflight_one_dataset(name)
        print(f"  [preflight] {name} OK ({time.time()-t_ds:.1f}s)", flush=True)
    print(f"\n  Preflight done in {(time.time()-t0)/60:.1f} min.\n")


def train(train_loader, model, optimizer, evaluator, device,
          momentum_weight, sharp=None, criterion_type=0):
    criterion = nn.SmoothL1Loss(beta=0.5)
    scaler    = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    step_losses, num_targets = [], []
    for data in train_loader:
        if model.use_lap:
            b_pe = data.lap_pos_enc
            sf = torch.rand(b_pe.size(1)); sf[sf >= 0.5] = 1.0; sf[sf < 0.5] = -1.0
            data.lap_pos_enc = b_pe * sf.unsqueeze(0)
        data = data.to(device); optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=USE_AMP, dtype=AMP_DTYPE):
            loss, num_t = _compute_loss(model, data, criterion, criterion_type)
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        step_losses.append(loss.item()); num_targets.append(num_t)
        with torch.no_grad(): _ema_update(model, momentum_weight)
    return None, np.average(step_losses, weights=num_targets)


@torch.no_grad()
def test(loader, model, evaluator, device, criterion_type=0):
    criterion = nn.SmoothL1Loss(beta=0.5)
    step_losses, num_targets = [], []
    for data in loader:
        data = data.to(device)
        with torch.amp.autocast('cuda', enabled=USE_AMP, dtype=AMP_DTYPE):
            loss, num_t = _compute_loss(model, data, criterion, criterion_type)
        step_losses.append(loss.item()); num_targets.append(num_t)
    return None, np.average(step_losses, weights=num_targets)


if __name__ == "__main__":
    skip_pf = os.environ.get("EXP12_SKIP_PREFLIGHT", "0") == "1"
    pf_only = os.environ.get("EXP12_PREFLIGHT_ONLY", "0") == "1"
    print(f"\n  EXP 12: Mamba-SSM (S6 selective scan) patch encoder")
    print(f"  Replaces ALL 6 Hadamard/Standard Transformer blocks with SSMEncoder")

    if not skip_pf:
        try: run_preflight_all()
        except Exception as e: print(f"\n  PREFLIGHT FAILED: {e!r}"); raise
        if pf_only: sys.exit(0)

    wall_times = {}; total_start = time.time()
    for dataset_name in DATASETS_TO_RUN:
        ds_start = time.time()
        print(f"\n{'#'*70}\n#  EXP 12 — SSM — DATASET: {dataset_name}\n{'#'*70}")
        updated_cfg = _merge_cfg_for_dataset(dataset_name)
        is_regression = (dataset_name == "ZINC")
        if is_regression:
            run(updated_cfg, create_dataset, create_model_ssm, train, test)
        else:
            run_k_fold(updated_cfg, create_dataset, create_model_ssm, train, test)
        ds_elapsed = time.time() - ds_start
        wall_times[dataset_name] = ds_elapsed
        print(f"\n  [{dataset_name}] wall time: {ds_elapsed/60:.1f} min")

    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}\n  EXP 12 (Mamba-SSM) COMPLETE | Total: {total_elapsed/60:.1f} min\n{'='*70}")
    for ds, t in wall_times.items(): print(f"  {ds:<12}  {t/60:>9.1f}m")
    print("  [TRACKER] Copy results above → EXP 12 row in tracker.md")
