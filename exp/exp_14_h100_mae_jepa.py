# ============================================================
# EXP 14 — HMS-JEPA + Hybrid MAE Reconstruction + JEPA Dual Objective
# ============================================================
# Novelty: Combine JEPA predictive objective (predict target patch
# representation) with Masked Auto-Encoder (MAE) generative reconstruction
# (reconstruct original node features of masked patches). These two
# objectives are theoretically COMPLEMENTARY:
#   - JEPA: learn structural/topological abstractions (representation-level)
#   - MAE:  learn fine-grained feature details (feature-level)
#
# This dual pretraining objective is inspired by recent HYBRID SSL works
# and directly addresses a known weakness of pure JEPA: the latent
# prediction loss may learn trivial / collapsed representations for
# datasets with uninformative node features (IMDB-B/M have NO node feats).
#
# For IMDB datasets (no node features): MAE loss is disabled automatically.
# For MUTAG / PROTEINS / DD: MAE operates on initial node features.
#
# Architecture additions (all self-contained):
#   • PatchDecoder: 3-layer MLP that reconstructs mean node features
#     of a patch from its context-encoded representation.
#   • MAE training: mask ratio = 0.5 (different from JEPA masks),
#     reconstruction target = mean of original node feature vectors.
#
# Theoretical grounding:
#   - He et al., "Masked Autoencoders Are Scalable Vision Learners",
#     CVPR 2022. (MAE)
#   - Hou et al., "GraphMAE: Self-Supervised Masked Graph Autoencoders",
#     KDD 2022. (MAE for graphs)
#   - Zhang et al., "GraphMAE2: A Decoding-Enhanced Masked Self-Supervised
#     Graph Learner", WWW 2023. (masked + predictive hybrid)
#   - Chen et al., "Context Autoencoder for Self-Supervised Representation
#     Learning", IJCV 2023 / ICCV 2023. (MAE + JEPA style hybrid)
#   - Dong et al., "MGAE: Masked Autoencoders with Self-Supervised
#     Explicit and Implicit Pretraining", ICML 2024.
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
from train.zinc import _ema_update

torch.backends.cudnn.benchmark = True
USE_AMP   = torch.cuda.is_available()
AMP_DTYPE = torch.bfloat16

# ─────────────────────────────────────────────────────────────
# HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────
MAE_MASK_RATIO = 0.5     # fraction of patches to reconstruct (separate from JEPA masks)
MAE_LOSS_WEIGHT = 0.3    # weight of MAE loss relative to JEPA loss
# Datasets with NO node features → disable MAE
DATASETS_WITHOUT_NODE_FEATS = {"IMDB-BINARY", "IMDB-MULTI"}

# ─────────────────────────────────────────────────────────────
# PATCH DECODER (MAE reconstruction head)
# ─────────────────────────────────────────────────────────────
class PatchDecoder(nn.Module):
    """
    Decodes a patch embedding → reconstructed mean node feature vector.
    Input:  [B*nMask, nhid]  (masked patch embeddings from context encoder)
    Output: [B*nMask, nfeat] (reconstructed original node features)
    """
    def __init__(self, nhid: int, nfeat_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(nhid),
            nn.Linear(nhid, nhid),
            nn.GELU(),
            nn.Linear(nhid, nhid // 2),
            nn.GELU(),
            nn.Linear(nhid // 2, nfeat_out),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────
# HMS-JEPA + MAE DUAL OBJECTIVE (self-contained subclass)
# ─────────────────────────────────────────────────────────────
class GraphHMSJepaMAE(GraphHMSJepa):
    """
    HMS-JEPA augmented with an MAE reconstruction decoder.
    The MAE head operates on L0 patches: given context encoder outputs
    for randomly MASKED patches, reconstruct their original mean node features.

    Constructor adds:
        nfeat_node_raw: original node feature dimension (before input_encoder)
        mae_enabled:    whether to apply MAE (False for IMDB-type datasets)
    """
    def __init__(self, *args, nfeat_node_raw=1, mae_enabled=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.mae_enabled   = mae_enabled
        self.nfeat_node_raw = nfeat_node_raw
        if mae_enabled:
            self.patch_decoder = PatchDecoder(self.nhid, nfeat_node_raw)
            # Learnable mask token (replaces masked patch embedding in decoder input)
            self.mask_token = nn.Parameter(torch.zeros(1, self.nhid))
            nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward_mae(self, data):
        """
        Run the standard HMS-JEPA forward, then compute MAE reconstruction
        for a random subset of L0 patches.

        Returns:
            jepa_output: standard ((tx0,ty0), (tx1,ty1), (tx2,ty2))
            mae_loss:    scalar (0.0 if mae_enabled=False or no node feats)
        """
        device = data.y.device

        # ── Standard JEPA forward ─────────────────────────────────────────
        jepa_output = super().forward(data)

        # ── MAE auxiliary ─────────────────────────────────────────────────
        if not self.mae_enabled:
            return jepa_output, torch.tensor(0.0, device=device)

        # Re-run GNN backbone to get raw patch embeddings (before JEPA masking)
        subgraph_x_L0, raw_patch_pes, batch_x = self._gnn_forward(data)
        # Add patch PE
        subgraph_x_L0_pe = subgraph_x_L0 + self.patch_rw_encoder(raw_patch_pes)

        # Build [B, n_patches, nhid] (fixed n_patches = call_n_patches[0][0])
        B         = len(data.call_n_patches)
        n_patches = data.call_n_patches[0][0]

        # Group patches by graph
        # subgraph_x_L0_pe is [total_patches, nhid]; reshape to [B, n_patches, nhid]
        try:
            patch_grid = subgraph_x_L0_pe.reshape(B, n_patches, self.nhid)
        except RuntimeError:
            # Fallback: not all graphs have same n_patches, skip MAE
            return jepa_output, torch.tensor(0.0, device=device)

        # Random MAE mask
        n_mask    = max(1, int(n_patches * MAE_MASK_RATIO))
        rand_idx  = torch.rand(B, n_patches, device=device).argsort(dim=-1)
        mask_idx  = rand_idx[:, :n_mask]          # [B, n_mask] — patch indices to mask
        keep_idx  = rand_idx[:, n_mask:]           # [B, n_keep]

        # Build decoder input: replace masked patches with mask_token
        decoder_x = patch_grid.clone()
        # Scatter mask tokens
        mask_token_expanded = self.mask_token.expand(B, n_mask, -1)  # [B, n_mask, nhid]
        decoder_x.scatter_(1, mask_idx.unsqueeze(-1).expand(-1,-1,self.nhid), mask_token_expanded)

        # Pass through context encoder (full sequence with mask tokens)
        ctx_mask_full = data.mask if hasattr(data, 'mask') else None
        decoder_out = self.context_encoder_L0(
            decoder_x,
            data.coarsen_adj if hasattr(data, 'coarsen_adj') else None,
            None
        )  # [B, n_patches, nhid]

        # Extract decoder output only at masked positions
        masked_out = decoder_out.gather(
            1, mask_idx.unsqueeze(-1).expand(-1,-1,self.nhid)
        ).reshape(B * n_mask, self.nhid)  # [B*n_mask, nhid]

        # Decode to original node features
        recon = self.patch_decoder(masked_out)  # [B*n_mask, nfeat_node_raw]

        # Compute reconstruction target: mean original node features per patch
        # We need per-patch mean of data.x (before input_encoder)
        # data.x: [total_nodes, nfeat_node_raw]
        # data.subgraphs_nodes_mapper: node → subgraph (L0 patch) mapping
        # subgraphs_batch: subgraph → batch index
        raw_x = data.x.float()  # [total_nodes, nfeat_node_raw]
        # Compute per-patch mean of raw node features
        patch_mean_feat = scatter(
            raw_x[data.subgraphs_nodes_mapper],
            data.subgraphs_batch,
            dim=0, reduce='mean'
        )  # [total_patches, nfeat_node_raw]

        # Reshape to [B, n_patches, nfeat_node_raw]
        try:
            patch_feat_grid = patch_mean_feat.reshape(B, n_patches, -1)
        except RuntimeError:
            return jepa_output, torch.tensor(0.0, device=device)

        # Extract targets at masked positions
        nfeat = patch_feat_grid.shape[-1]
        masked_targets = patch_feat_grid.gather(
            1, mask_idx.unsqueeze(-1).expand(-1,-1,nfeat)
        ).reshape(B * n_mask, nfeat)  # [B*n_mask, nfeat_node_raw]

        # Handle dimension mismatch (if nfeat != nfeat_node_raw)
        if recon.shape[-1] != masked_targets.shape[-1]:
            return jepa_output, torch.tensor(0.0, device=device)

        mae_loss = F.mse_loss(recon, masked_targets.detach())
        return jepa_output, mae_loss


# ─────────────────────────────────────────────────────────────
# MODEL FACTORY
# ─────────────────────────────────────────────────────────────
_CURRENT_DATASET = ["MUTAG"]

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


def create_model_mae(cfg):
    base = create_model(cfg)
    if not isinstance(base, GraphHMSJepa):
        return base

    ds = _CURRENT_DATASET[0]
    mae_enabled = ds not in DATASETS_WITHOUT_NODE_FEATS
    node_type, edge_type, nfeat_node, nfeat_edge = _resolve_dataset_types(ds)

    # Try to determine nfeat_node_raw from dataset name
    NFEAT_RAW = {"MUTAG": 7, "PROTEINS": 3, "DD": 89, "IMDB-BINARY": 0, "IMDB-MULTI": 0}
    nfeat_node_raw = NFEAT_RAW.get(ds, 1)
    if nfeat_node_raw == 0:
        mae_enabled = False

    model = GraphHMSJepaMAE(
        nfeat_node=nfeat_node, nfeat_edge=nfeat_edge,
        nhid=base.nhid, nout=1,
        nlayer_gnn=len(base.gnns), nlayer_mlpmixer=cfg.model.nlayer_mlpmixer,
        node_type=node_type, edge_type=edge_type,
        gnn_type=cfg.model.gnn_type, gMHA_type=cfg.model.gMHA_type,
        rw_dim=cfg.pos_enc.rw_dim, lap_dim=cfg.pos_enc.lap_dim,
        dropout=getattr(cfg.train,'dropout',0), mlpmixer_dropout=getattr(cfg.train,'mlpmixer_dropout',0),
        n_patches=cfg.metis.n_patches, patch_rw_dim=cfg.pos_enc.patch_rw_dim,
        num_context_patches=cfg.jepa.num_context, num_target_patches=cfg.jepa.num_targets,
        num_target_patches_L1=cfg.jepa.num_targets_L1, num_target_patches_L2=cfg.jepa.num_targets_L2,
        loss_weights=cfg.jepa.loss_weights, var_weight=cfg.jepa.var_weight,
        nfeat_node_raw=nfeat_node_raw, mae_enabled=mae_enabled,
    )
    print(f"  [MAE] dataset={ds}  mae_enabled={mae_enabled}  nfeat_node_raw={nfeat_node_raw}"
          f"  mae_mask_ratio={MAE_MASK_RATIO}  mae_weight={MAE_LOSS_WEIGHT}")
    return model


# ─────────────────────────────────────────────────────────────
# LOSS
# ─────────────────────────────────────────────────────────────
def _mae_jepa_loss(model, data):
    """Combined JEPA + MAE loss."""
    if not isinstance(model, GraphHMSJepaMAE):
        from train.zinc import _compute_loss
        return _compute_loss(model, data, nn.SmoothL1Loss(beta=0.5), 0)

    criterion = nn.SmoothL1Loss(beta=0.5)
    jepa_out, mae_loss = model.forward_mae(data)
    (tx0, ty0), (tx1, ty1), (tx2, ty2) = jepa_out

    l0 = criterion(ty0, tx0); l1 = criterion(ty1, tx1); l2 = criterion(ty2, tx2)
    w  = model.loss_weights
    jepa_loss = w[0]*l0 + w[1]*l1 + w[2]*l2
    jepa_loss = jepa_loss + model.var_weight * torch.mean(torch.relu(1.0 - tx0.detach().std(dim=0)))

    total = jepa_loss + MAE_LOSS_WEIGHT * mae_loss
    return total, len(ty0)


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
    cfg_path = f"/tmp/exp14_{dataset_name.replace('-','_').lower()}.yaml"
    with open(cfg_path, "w") as f: f.write(DATASET_CONFIGS[dataset_name])
    from core.config import cfg as _cfg
    _cfg.defrost(); _cfg.merge_from_file(cfg_path)
    updated = update_cfg(_cfg, args_str=""); updated.k = 10
    return updated

def preflight_one_dataset(dataset_name):
    _CURRENT_DATASET[0] = dataset_name
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
        train_indices, _ = k_fold(dataset, cfg.k); ti = train_indices[0]
        n = min(_PREFLIGHT_MAX_GRAPHS, len(ti))
        subset = dataset[ti[:n]]; subset.transform = transform
        train_list = [x for x in subset]
        bs = min(cfg.train.batch_size, max(1, len(train_list)))
        loader = DataLoader(train_list, batch_size=bs, shuffle=True, num_workers=0, pin_memory=False)
    model = create_model_mae(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    data = next(iter(loader)).to(device); optimizer.zero_grad()
    with torch.amp.autocast("cuda", enabled=USE_AMP, dtype=AMP_DTYPE):
        loss, _ = _mae_jepa_loss(model, data)
    scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()

def run_preflight_all():
    print("\n" + "="*70)
    print("  EXP 14 — PREFLIGHT (JEPA + MAE Dual Objective)")
    print("="*70)
    t0 = time.time()
    for name in DATASETS_TO_RUN:
        _CURRENT_DATASET[0] = name
        t_ds = time.time()
        print(f"\n  [preflight] {name} ...", flush=True)
        preflight_one_dataset(name)
        print(f"  [preflight] {name} OK ({time.time()-t_ds:.1f}s)", flush=True)
    print(f"\n  Preflight done in {(time.time()-t0)/60:.1f} min.\n")

def train(train_loader, model, optimizer, evaluator, device,
          momentum_weight, sharp=None, criterion_type=0):
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    step_losses, num_targets = [], []
    for data in train_loader:
        if model.use_lap:
            b_pe = data.lap_pos_enc
            sf = torch.rand(b_pe.size(1)); sf[sf >= 0.5] = 1.0; sf[sf < 0.5] = -1.0
            data.lap_pos_enc = b_pe * sf.unsqueeze(0)
        data = data.to(device); optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=USE_AMP, dtype=AMP_DTYPE):
            loss, num_t = _mae_jepa_loss(model, data)
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        step_losses.append(loss.item()); num_targets.append(num_t)
        with torch.no_grad(): _ema_update(model, momentum_weight)
    return None, np.average(step_losses, weights=num_targets)

@torch.no_grad()
def test(loader, model, evaluator, device, criterion_type=0):
    step_losses, num_targets = [], []
    for data in loader:
        data = data.to(device)
        with torch.amp.autocast('cuda', enabled=USE_AMP, dtype=AMP_DTYPE):
            loss, num_t = _mae_jepa_loss(model, data)
        step_losses.append(loss.item()); num_targets.append(num_t)
    return None, np.average(step_losses, weights=num_targets)

if __name__ == "__main__":
    skip_pf = os.environ.get("EXP14_SKIP_PREFLIGHT", "0") == "1"
    pf_only = os.environ.get("EXP14_PREFLIGHT_ONLY", "0") == "1"
    print(f"\n  EXP 14: Hybrid JEPA + MAE Dual Objective")
    print(f"  MAE mask_ratio={MAE_MASK_RATIO}  mae_weight={MAE_LOSS_WEIGHT}")
    print(f"  Datasets without node feats (MAE disabled): {DATASETS_WITHOUT_NODE_FEATS}")

    if not skip_pf:
        try: run_preflight_all()
        except Exception as e: print(f"\n  PREFLIGHT FAILED: {e!r}"); raise
        if pf_only: sys.exit(0)

    wall_times = {}; total_start = time.time()
    for dataset_name in DATASETS_TO_RUN:
        _CURRENT_DATASET[0] = dataset_name
        ds_start = time.time()
        print(f"\n{'#'*70}\n#  EXP 14 — JEPA+MAE — DATASET: {dataset_name}\n{'#'*70}")
        updated_cfg = _merge_cfg_for_dataset(dataset_name)
        is_regression = (dataset_name == "ZINC")
        if is_regression:
            run(updated_cfg, create_dataset, create_model_mae, train, test)
        else:
            run_k_fold(updated_cfg, create_dataset, create_model_mae, train, test)
        ds_elapsed = time.time() - ds_start
        wall_times[dataset_name] = ds_elapsed
        print(f"\n  [{dataset_name}] wall time: {ds_elapsed/60:.1f} min")

    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}\n  EXP 14 (JEPA+MAE) COMPLETE | Total: {total_elapsed/60:.1f} min\n{'='*70}")
    for ds, t in wall_times.items(): print(f"  {ds:<12}  {t/60:>9.1f}m")
    print("  [TRACKER] Copy results above → EXP 14 row in tracker.md")
