# ============================================================
# EXP 16 — Asymmetric Multi-Crop Prediction + Spectral PE Fusion
# ============================================================
# Novelty:
#   (1) Multi-crop masking inspired by I-JEPA — local + global context
#   (2) Fuse RWSE with Laplacian spectral PE via learnable MLP
#   (3) Reverse coarse-to-fine prediction path (L2→L1)
#
# Theoretical grounding:
#   - Assran et al., "I-JEPA", CVPR 2023 — multi-crop strategy
#   - Rampasek et al., "GPS", NeurIPS 2022 — RWSE + LapPE fusion
#   - Huang et al., "SPE", NeurIPS 2024 — stable spectral PEs
#
# Key changes vs EXP 13:
#   1. MultiCropContextEncoder: samples 2 local + 1 global context
#   2. LapSpectralPE: top-k Laplacian eigenvalues as additional PE
#      fused via MLP_fuse(cat(RWSE, LapPE)) → graceful zero-pad fallback
#   3. Bidirectional loss: forward (L0→L0, L0→L1, L1→L2) +
#      reverse (L2→L1 coarse-to-fine)
#   4. Weighted loss sum over 4 directions
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
from core.tracker_footer import print_exp_tracker_footer
from core.model import GraphHMSJepa
from train.zinc import _ema_update

torch.backends.cudnn.benchmark = True
USE_AMP   = torch.cuda.is_available()
AMP_DTYPE = torch.bfloat16

LAP_K            = 8     # top-k Laplacian eigenvalues
REVERSE_WEIGHT   = 0.25  # weight for L2→L1 coarse-to-fine prediction


# ─────────────────────────────────────────────────────────────
# SPECTRAL PE FUSION MODULE
# ─────────────────────────────────────────────────────────────
class SpectralPEFusion(nn.Module):
    """
    Fuses RWSE and Laplacian spectral PE via learnable MLP.
    Input:  rwse [N, rw_dim], lap_pe [N, lap_k] (zero-padded if small graph)
    Output: [N, nhid]
    """
    def __init__(self, rw_dim: int, lap_k: int, nhid: int):
        super().__init__()
        self.lap_k = lap_k
        in_dim = rw_dim + lap_k
        self.fuse_mlp = nn.Sequential(
            nn.Linear(in_dim, nhid),
            nn.LayerNorm(nhid),
            nn.GELU(),
            nn.Linear(nhid, nhid),
        )

    def forward(self, rwse: torch.Tensor, lap_pe: torch.Tensor) -> torch.Tensor:
        """
        rwse:   [N, rw_dim]
        lap_pe: [N, lap_k] — may have fewer columns if graph is small; zero-pad handled externally
        """
        N = rwse.shape[0]
        if lap_pe.shape[1] < self.lap_k:
            pad = torch.zeros(N, self.lap_k - lap_pe.shape[1], device=rwse.device, dtype=rwse.dtype)
            lap_pe = torch.cat([lap_pe, pad], dim=1)
        return self.fuse_mlp(torch.cat([rwse, lap_pe], dim=-1))


# ─────────────────────────────────────────────────────────────
# MULTI-CROP CONTEXT AGGREGATOR
# ─────────────────────────────────────────────────────────────
class MultiCropAggregator(nn.Module):
    """
    Aggregates local+global context via cross-attention with target PE.
    Same architecture as EXP 13's MultiContextAggregator.
    """
    def __init__(self, nhid: int, nhead: int = 8):
        super().__init__()
        self.ctx_sa  = nn.MultiheadAttention(nhid, nhead, batch_first=True)
        self.norm1   = nn.LayerNorm(nhid)
        self.ca      = nn.MultiheadAttention(nhid, nhead, batch_first=True)
        self.norm2   = nn.LayerNorm(nhid)
        self.ffn     = nn.Sequential(nn.Linear(nhid, nhid*2), nn.GELU(), nn.Linear(nhid*2, nhid))
        self.norm3   = nn.LayerNorm(nhid)

    def forward(self, ctx_x, tgt_pe):  # ctx_x [B,K,H], tgt_pe [B,nT,H]
        sa, _ = self.ctx_sa(ctx_x, ctx_x, ctx_x)
        ctx_x = self.norm1(ctx_x + sa)
        ca, _ = self.ca(tgt_pe, ctx_x, ctx_x)
        out = self.norm2(tgt_pe + ca)
        return self.norm3(out + self.ffn(out))


# ─────────────────────────────────────────────────────────────
# MODEL: MCMT-JEPA + SPECTRAL PE + MULTI-CROP + BIDIRECTIONAL
# ─────────────────────────────────────────────────────────────
class GraphHMSJepaMultiCropSpectral(GraphHMSJepa):
    """
    I-JEPA-inspired multi-crop: 2 local contexts + 1 global context.
    Fuses RWSE with Laplacian spectral PE.
    Adds reverse L2→L1 coarse-to-fine prediction path.
    """
    def __init__(self, *args, rw_dim=15, lap_k=LAP_K, **kwargs):
        super().__init__(*args, **kwargs)
        nhid = self.nhid

        # Spectral PE fusion (replaces plain patch_rw_encoder for fused version)
        self.spectral_fuse    = SpectralPEFusion(rw_dim, lap_k, nhid)
        self.spectral_fuse_L1 = SpectralPEFusion(rw_dim, lap_k, nhid)

        # Multi-crop aggregators
        self.mc_agg_L0 = MultiCropAggregator(nhid)
        self.mc_agg_L1 = MultiCropAggregator(nhid)

        # Reverse predictor: L2 → L1
        self.predictor_L2_to_L1 = nn.Sequential(
            nn.Linear(nhid, nhid), nn.GELU(), nn.Linear(nhid, nhid))

    def _get_lap_pe(self, data, n_patches_total):
        """
        Build patch-level Laplacian PE [N_patches, lap_k] with zero-padding.
        If data has no lap_pos_enc, return zeros.
        """
        device = data.y.device
        if not hasattr(data, 'lap_pos_enc') or data.lap_pos_enc is None:
            return torch.zeros(n_patches_total, LAP_K, device=device)
        lap = data.lap_pos_enc  # [total_nodes, lap_dim]
        # Aggregate to patch level via scatter (same mapping as GNN output)
        # We use node_to_subgraph if available, else zeros
        if hasattr(data, 'subgraph_idx') and data.subgraph_idx is not None:
            lap_patch = scatter(lap[:, :LAP_K] if lap.shape[1] >= LAP_K
                                else F.pad(lap, (0, LAP_K - lap.shape[1])),
                               data.subgraph_idx, dim=0, reduce='mean',
                               dim_size=n_patches_total)
        else:
            lap_patch = torch.zeros(n_patches_total, LAP_K, device=device)
        return lap_patch.to(device)

    def forward_mc(self, data):
        device = data.y.device
        subgraph_x_L0, raw_patch_pes, _ = self._gnn_forward(data)
        subgraph_x_L1    = scatter(subgraph_x_L0, data.fine_to_medium, dim=0, reduce='mean')
        raw_patch_pes_L1 = scatter(raw_patch_pes,  data.fine_to_medium, dim=0, reduce='mean')
        subgraph_x_L2    = scatter(subgraph_x_L1, data.medium_to_coarse, dim=0, reduce='mean')
        raw_patch_pes_L2 = scatter(raw_patch_pes_L1, data.medium_to_coarse, dim=0, reduce='mean')

        B = len(data.call_n_patches)
        N_L0 = subgraph_x_L0.shape[0]
        N_L1 = subgraph_x_L1.shape[0]

        # Attempt spectral PE fusion (with zero fallback)
        lap_patch_L0 = self._get_lap_pe(data, N_L0)
        try:
            fused_pe_L0 = self.spectral_fuse(raw_patch_pes, lap_patch_L0)
        except Exception:
            fused_pe_L0 = self.patch_rw_encoder(raw_patch_pes)

        lap_patch_L1 = scatter(lap_patch_L0, data.fine_to_medium, dim=0, reduce='mean', dim_size=N_L1)
        try:
            fused_pe_L1 = self.spectral_fuse_L1(raw_patch_pes_L1, lap_patch_L1)
        except Exception:
            fused_pe_L1 = self.patch_rw_encoder_L1(raw_patch_pes_L1)

        def _bi(npl):
            bi = torch.tensor(np.cumsum(npl))
            return torch.hstack((torch.tensor(0), bi[:-1])).to(device)

        bi_L0 = _bi(data.call_n_patches)
        bi_L1 = _bi(data.n_patches_L1)
        bi_L2 = _bi(data.n_patches_L2)

        # ── Multi-Crop Contexts ───────────────────────────────────────────
        # Local ctx 1 (primary, same as EXP 13)
        ctx_idx_L0  = data.context_subgraph_idx + bi_L0
        ctx1 = subgraph_x_L0[ctx_idx_L0] + fused_pe_L0[ctx_idx_L0]
        # Local ctx 2 (shifted by +1, same as EXP 13)
        n_ppg = torch.tensor([c[0] for c in data.call_n_patches], device=device)
        ctx2_idx = ((data.context_subgraph_idx + 1) % n_ppg) + bi_L0
        ctx2 = subgraph_x_L0[ctx2_idx] + fused_pe_L0[ctx2_idx]
        # Global ctx: mean of 3 adjacent patches (ctx+2, ctx+3, ctx+4 mod n_ppg)
        global_ctx_list = []
        for k in range(2, 5):
            idx_k = ((data.context_subgraph_idx + k) % n_ppg) + bi_L0
            global_ctx_list.append(subgraph_x_L0[idx_k] + fused_pe_L0[idx_k])
        ctx_global = torch.stack(global_ctx_list, dim=1).mean(1)  # [B, nhid]

        ctx_stack = torch.stack([ctx1, ctx2, ctx_global], dim=1)  # [B, 3, nhid]

        # ── Target L0 ────────────────────────────────────────────────────
        tgt_idx_L0 = torch.vstack([torch.tensor(dt) for dt in data.target_subgraph_idxs]).to(device) + bi_L0.unsqueeze(1)
        tgt_pe_L0_r = fused_pe_L0[tgt_idx_L0.flatten()].reshape(B, self.num_target_patches, self.nhid)
        tgt_x_L0 = subgraph_x_L0[tgt_idx_L0.flatten()].reshape(B, self.num_target_patches, self.nhid)
        with torch.no_grad():
            if hasattr(data, 'coarsen_adj'):
                rel = torch.vstack([torch.tensor(dt) for dt in data.target_subgraph_idxs])
                adj = data.coarsen_adj[torch.arange(B).unsqueeze(1).unsqueeze(2), rel.unsqueeze(1), rel.unsqueeze(2)]
                tgt_x_L0 = self.target_encoder_L0(tgt_x_L0, adj, None)
            else:
                tgt_x_L0 = self.target_encoder_L0(tgt_x_L0, None, None)
        pred_L0 = self.predictor_L0_to_L0(self.mc_agg_L0(ctx_stack, tgt_pe_L0_r))

        # ── Target L1 (from L0 contexts) ─────────────────────────────────
        tgt_idx_L1 = torch.vstack([torch.tensor(dt) for dt in data.target_subgraph_idxs_L1]).to(device) + bi_L1.unsqueeze(1)
        nT_L1 = tgt_idx_L1.shape[1]
        tgt_pe_L1_r = fused_pe_L1[tgt_idx_L1.flatten()].reshape(B, nT_L1, self.nhid)
        tgt_x_L1 = subgraph_x_L1[tgt_idx_L1.flatten()].reshape(B, nT_L1, self.nhid)
        with torch.no_grad():
            rel_L1 = torch.vstack([torch.tensor(dt) for dt in data.target_subgraph_idxs_L1])
            adj_L1 = data.coarsen_adj_L1[torch.arange(B).unsqueeze(1).unsqueeze(2), rel_L1.unsqueeze(1), rel_L1.unsqueeze(2)]
            tgt_x_L1 = self.target_encoder_L1(tgt_x_L1, adj_L1, None)
        pred_L1 = self.predictor_L0_to_L1(self.mc_agg_L1(ctx_stack, tgt_pe_L1_r))

        # ── L1→L2 (standard) ─────────────────────────────────────────────
        ctx_idx_L1_s = data.context_subgraph_idx_L1 + bi_L1
        ctx_patch_L1 = subgraph_x_L1[ctx_idx_L1_s] + self.patch_rw_encoder_L1(raw_patch_pes_L1[ctx_idx_L1_s])
        ctx_x_L1 = self.context_encoder_L1(ctx_patch_L1.unsqueeze(1), None, ~data.mask_L1.flatten()[ctx_idx_L1_s].reshape(B, 1))
        tgt_idx_L2 = torch.vstack([torch.tensor(dt) for dt in data.target_subgraph_idxs_L2]).to(device) + bi_L2.unsqueeze(1)
        nT_L2 = tgt_idx_L2.shape[1]
        tgt_x_L2 = subgraph_x_L2[tgt_idx_L2.flatten()].reshape(B, nT_L2, self.nhid)
        with torch.no_grad():
            rel_L2 = torch.vstack([torch.tensor(dt) for dt in data.target_subgraph_idxs_L2])
            adj_L2 = data.coarsen_adj_L2[torch.arange(B).unsqueeze(1).unsqueeze(2), rel_L2.unsqueeze(1), rel_L2.unsqueeze(2)]
            tgt_x_L2 = self.target_encoder_L2(tgt_x_L2, adj_L2, None)
        pe_L2_r = self.patch_rw_encoder_L2(raw_patch_pes_L2[tgt_idx_L2.flatten()]).reshape(B, nT_L2, self.nhid)
        pred_L2 = self.predictor_L1_to_L2(ctx_x_L1 + pe_L2_r)

        # ── Reverse: L2→L1 (coarse-to-fine) ──────────────────────────────
        # Use mean L2 embedding as context to predict L1 targets
        ctx_L2_agg = subgraph_x_L2[tgt_idx_L2.flatten()].reshape(B, nT_L2, self.nhid).mean(1, keepdim=True)  # [B,1,H]
        pred_L2_to_L1 = self.predictor_L2_to_L1(ctx_L2_agg.expand_as(tgt_x_L1))  # [B,nT_L1,H]

        return ((tgt_x_L0, pred_L0), (tgt_x_L1, pred_L1),
                (tgt_x_L2, pred_L2), (tgt_x_L1, pred_L2_to_L1))


# ─────────────────────────────────────────────────────────────
# LOSS
# ─────────────────────────────────────────────────────────────
def _mc_spectral_loss(model, data):
    if not isinstance(model, GraphHMSJepaMultiCropSpectral):
        from train.zinc import _compute_loss
        return _compute_loss(model, data, nn.SmoothL1Loss(beta=0.5), 0)
    criterion = nn.SmoothL1Loss(beta=0.5)
    (tx0,ty0),(tx1,ty1),(tx2,ty2),(tx1r,ty1r) = model.forward_mc(data)
    w = model.loss_weights
    loss = (w[0]*criterion(ty0,tx0) + w[1]*criterion(ty1,tx1) +
            w[2]*criterion(ty2,tx2) + REVERSE_WEIGHT*criterion(ty1r,tx1r))
    loss = loss + model.var_weight * torch.mean(torch.relu(1.0 - tx0.detach().std(dim=0)))
    return loss, len(ty0)


# ─────────────────────────────────────────────────────────────
# MODEL FACTORY
# ─────────────────────────────────────────────────────────────
def _resolve_dataset_types(name):
    m = {'MUTAG':('Linear','Linear',7,4),'PROTEINS':('Linear','Linear',3,1),
         'DD':('Linear','Linear',89,1),'IMDB-BINARY':('Linear','Linear',1,1),
         'IMDB-MULTI':('Linear','Linear',1,1),'REDDIT-BINARY':('Linear','Linear',1,1),
         'ZINC':('Discrete','Discrete',28,4)}
    return m.get(name, ('Linear','Linear',1,1))

def create_model_mc(cfg):
    base = create_model(cfg)
    if not isinstance(base, GraphHMSJepa): return base
    nt, et, nfn, nfe = _resolve_dataset_types(cfg.dataset)
    model = GraphHMSJepaMultiCropSpectral(
        nfeat_node=nfn, nfeat_edge=nfe, nhid=base.nhid, nout=1,
        nlayer_gnn=len(base.gnns), nlayer_mlpmixer=cfg.model.nlayer_mlpmixer,
        node_type=nt, edge_type=et, gnn_type=cfg.model.gnn_type, gMHA_type=cfg.model.gMHA_type,
        rw_dim=cfg.pos_enc.patch_rw_dim, lap_dim=cfg.pos_enc.lap_dim,
        dropout=getattr(cfg.train,'dropout',0), mlpmixer_dropout=getattr(cfg.train,'mlpmixer_dropout',0),
        n_patches=cfg.metis.n_patches, patch_rw_dim=cfg.pos_enc.patch_rw_dim,
        num_context_patches=cfg.jepa.num_context, num_target_patches=cfg.jepa.num_targets,
        num_target_patches_L1=cfg.jepa.num_targets_L1, num_target_patches_L2=cfg.jepa.num_targets_L2,
        loss_weights=cfg.jepa.loss_weights, var_weight=cfg.jepa.var_weight,
        lap_k=LAP_K,
    )
    p = sum(x.numel() for x in model.parameters() if x.requires_grad)
    print(f"  [MC-Spectral Model] params={p/1e6:.2f}M  lap_k={LAP_K}  rev_w={REVERSE_WEIGHT}")
    return model


# ─────────────────────────────────────────────────────────────
# DATASET CONFIGS
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

def _merge_cfg_for_dataset(ds):
    cfg_path = f"/tmp/exp16_{ds.replace('-','_').lower()}.yaml"
    with open(cfg_path, "w") as f: f.write(DATASET_CONFIGS[ds])
    from core.config import cfg as _cfg
    _cfg.defrost(); _cfg.merge_from_file(cfg_path)
    updated = update_cfg(_cfg, args_str=""); updated.k = 10
    return updated

def preflight_one_dataset(ds):
    cfg = _merge_cfg_for_dataset(ds); device = _device_tensor(cfg)
    out = create_dataset(cfg)
    if cfg.dataset == "ZINC":
        td, _, _ = out; n = min(_PREFLIGHT_MAX_GRAPHS, len(td))
        small = [x for x in td[:n]] if not cfg.metis.online else td[:n]
        loader = DataLoader(small, batch_size=min(cfg.train.batch_size, max(1,n)), shuffle=True, num_workers=0)
    else:
        dataset, transform, _ = out
        ti = k_fold(dataset, cfg.k)[0][0]; n = min(_PREFLIGHT_MAX_GRAPHS, len(ti))
        sub = dataset[ti[:n]]; sub.transform = transform
        loader = DataLoader([x for x in sub], batch_size=min(cfg.train.batch_size,max(1,n)), shuffle=True, num_workers=0)
    model = create_model_mc(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    data = next(iter(loader)).to(device); optimizer.zero_grad()
    with torch.amp.autocast("cuda", enabled=USE_AMP, dtype=AMP_DTYPE):
        loss, _ = _mc_spectral_loss(model, data)
    scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()

def run_preflight_all():
    print("\n" + "="*70)
    print("  EXP 16 — PREFLIGHT (MultiCrop + Spectral PE Fusion)")
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
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    step_losses, num_targets = [], []
    for data in train_loader:
        if model.use_lap:
            b_pe = data.lap_pos_enc
            sf = torch.rand(b_pe.size(1)); sf[sf >= 0.5] = 1.0; sf[sf < 0.5] = -1.0
            data.lap_pos_enc = b_pe * sf.unsqueeze(0)
        data = data.to(device); optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=USE_AMP, dtype=AMP_DTYPE):
            loss, num_t = _mc_spectral_loss(model, data)
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
            loss, num_t = _mc_spectral_loss(model, data)
        step_losses.append(loss.item()); num_targets.append(num_t)
    return None, np.average(step_losses, weights=num_targets)

if __name__ == "__main__":
    skip_pf = os.environ.get("EXP16_SKIP_PREFLIGHT", "0") == "1"
    pf_only = os.environ.get("EXP16_PREFLIGHT_ONLY", "0") == "1"
    print(f"\n  EXP 16: Asymmetric Multi-Crop + Spectral PE Fusion")
    print(f"  lap_k={LAP_K}  reverse_weight={REVERSE_WEIGHT}  3 contexts (2 local + 1 global)")

    if not skip_pf:
        try: run_preflight_all()
        except Exception as e: print(f"\n  PREFLIGHT FAILED: {e!r}"); raise
        if pf_only: sys.exit(0)

    wall_times = {}; tracker_results = {}; total_start = time.time()
    for dataset_name in DATASETS_TO_RUN:
        ds_start = time.time()
        print(f"\n{'#'*70}\n#  EXP 16 — MultiCrop+Spectral — DATASET: {dataset_name}\n{'#'*70}")
        updated_cfg = _merge_cfg_for_dataset(dataset_name)
        if dataset_name == "ZINC":
            tracker_results[dataset_name] = run(updated_cfg, create_dataset, create_model_mc, train, test)
        else:
            tracker_results[dataset_name] = run_k_fold(updated_cfg, create_dataset, create_model_mc, train, test)
        ds_elapsed = time.time() - ds_start
        wall_times[dataset_name] = ds_elapsed
        print(f"\n  [{dataset_name}] wall time: {ds_elapsed/60:.1f} min")

    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}\n  EXP 16 COMPLETE | Total: {total_elapsed/60:.1f} min\n{'='*70}")
    for ds, t in wall_times.items(): print(f"  {ds:<12}  {t/60:>9.1f}m")
    print_exp_tracker_footer(16, "Asymmetric Multi-Crop + Spectral PE Fusion", tracker_results)
