# ============================================================
# EXP 13 — HMS-JEPA + Multi-Context Multi-Target (MCMT) JEPA
# ============================================================
# Novelty: Replace the single context → multiple targets prediction
# with a MULTI-CONTEXT architecture where K context patches
# collaboratively predict targets. Additionally, add RANDOM PATCH
# ORDER PERMUTATION to make the model order-invariant (critical for
# METIS partitioning which has arbitrary ordering).
#
# Also introduce: Graph-Level Contrastive Auxiliary Loss
# (contrast between two views of the same graph's L2 embedding)
# — a complementary objective to the JEPA pretraining loss.
#
# All model code is SELF-CONTAINED.
#
# Theoretical grounding:
#   - Assran et al., "Self-Supervised Learning from Images with a
#     Joint-Embedding Predictive Architecture (I-JEPA)", CVPR 2023
#     — single context, multiple targets. We extend to multi-context.
#   - Bardes et al., "MC-JEPA: A Joint-Embedding Predictive
#     Architecture for Self-Supervised Learning of Motion and Content",
#     ICML 2024 Workshop. — multi-context JEPA design.
#   - You et al., "Graph Contrastive Learning with Augmentations (GraphCL)",
#     NeurIPS 2020 — graph-level contrastive auxiliary loss.
#   - MolCA (NeurIPS 2023), Cross-Modal JEPA ideas.
#   - Lee et al., "Set Transformer: A Framework for Attention-based
#     Permutation-Invariant Neural Networks", ICML 2019
#     → multi-head attention aggregation for multi-context.
#
# Key design decisions:
#   1. NUM_CONTEXTS = 2 (use 2 context patches instead of 1)
#   2. Context aggregation via Cross-Attention (context_q, target_kv)
#      → more expressive than simple concatenation
#   3. Patch order permutation: randomly shuffle patch sequence each batch
#   4. Auxiliary graph-level NT-Xent loss (temperature=0.1) on L2 embeddings
#      of two random patches as graph-level "views"
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
from core.model_utils.elements import MLP
from train.zinc import _ema_update

torch.backends.cudnn.benchmark = True
USE_AMP   = torch.cuda.is_available()
AMP_DTYPE = torch.bfloat16

# ─────────────────────────────────────────────────────────────
# HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────
NUM_CONTEXTS        = 2       # number of context patches (vs original 1)
NT_XENT_WEIGHT      = 0.05   # weight for auxiliary contrastive loss
NT_XENT_TEMPERATURE = 0.1    # NT-Xent temperature
PERMUTE_PATCHES     = True   # randomly permute patch order each batch


# ─────────────────────────────────────────────────────────────
# MULTI-CONTEXT CROSS-ATTENTION MODULE
# ─────────────────────────────────────────────────────────────
class MultiContextAggregator(nn.Module):
    """
    Aggregates K context patch embeddings into a SINGLE context vector
    for each target. Uses a learned Set Transformer (ISAB-style) pooling
    followed by cross-attention with target PEs.

    Input:
      ctx_x:   [B, K, nhid]  (K context patches)
      tgt_pe:  [B, nT, nhid] (target positional encodings)
    Output:
      [B, nT, nhid]  (context embedding conditioned on target PE)
    """
    def __init__(self, nhid: int, nhead: int = 8, n_inducers: int = 4):
        super().__init__()
        self.nhid = nhid
        # Self-attention over context patches
        self.ctx_self_attn = nn.MultiheadAttention(nhid, nhead, batch_first=True, dropout=0.0)
        self.ctx_norm1     = nn.LayerNorm(nhid)
        # Cross-attention: target PE queries context
        self.cross_attn    = nn.MultiheadAttention(nhid, nhead, batch_first=True, dropout=0.0)
        self.cross_norm    = nn.LayerNorm(nhid)
        # FFN
        self.ffn  = nn.Sequential(nn.Linear(nhid, nhid*2), nn.GELU(), nn.Linear(nhid*2, nhid))
        self.norm2 = nn.LayerNorm(nhid)

    def forward(self, ctx_x, tgt_pe):
        """
        ctx_x:  [B, K, nhid]
        tgt_pe: [B, nT, nhid]
        returns [B, nT, nhid]
        """
        # Self-attention within contexts
        ctx_sa, _ = self.ctx_self_attn(ctx_x, ctx_x, ctx_x)
        ctx_x = self.ctx_norm1(ctx_x + ctx_sa)                   # [B, K, nhid]

        # Cross-attention: query=target_PE, key/value=context
        ctx_ca, _ = self.cross_attn(tgt_pe, ctx_x, ctx_x)        # [B, nT, nhid]
        out = self.cross_norm(tgt_pe + ctx_ca)
        out = self.norm2(out + self.ffn(out))
        return out


# ─────────────────────────────────────────────────────────────
# NT-XENT AUXILIARY CONTRASTIVE LOSS
# ─────────────────────────────────────────────────────────────
def nt_xent_loss(z1, z2, temperature=NT_XENT_TEMPERATURE):
    """
    NT-Xent (Normalised Temperature-scaled Cross Entropy) loss.
    z1, z2: [B, nhid]  — two views of the same graph
    """
    B = z1.shape[0]
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    z  = torch.cat([z1, z2], dim=0)                    # [2B, nhid]
    sim = torch.mm(z, z.T) / temperature                # [2B, 2B]
    # Remove self-similarity (diagonal)
    mask = torch.eye(2*B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float('-inf'))
    # Positive pairs: (i, i+B) and (i+B, i)
    labels = torch.cat([torch.arange(B, 2*B), torch.arange(B)]).to(z.device)
    return F.cross_entropy(sim, labels)


# ─────────────────────────────────────────────────────────────
# HMS-JEPA MCMT VARIANT (self-contained subclass)
# ─────────────────────────────────────────────────────────────
class GraphHMSJepaMCMT(GraphHMSJepa):
    """
    Multi-Context Multi-Target HMS-JEPA.
    Uses NUM_CONTEXTS=2 context patches and CrossAttention aggregation.
    Adds NT-Xent auxiliary contrastive loss at graph level.
    """
    def __init__(self, *args, num_contexts=2, **kwargs):
        # Must pass num_context_patches=1 to parent (we override context handling)
        super().__init__(*args, **kwargs)
        self.num_contexts = num_contexts
        self.nhid = kwargs.get('nhid', self.nhid)

        # Multi-context aggregators for L0 prediction
        self.mc_agg_L0 = MultiContextAggregator(self.nhid)
        self.mc_agg_L1 = MultiContextAggregator(self.nhid)

        # Projection head for NT-Xent (graph-level)
        self.proj_head = nn.Sequential(
            nn.Linear(self.nhid, self.nhid),
            nn.GELU(),
            nn.Linear(self.nhid, 128),
        )

    def forward_mcmt(self, data):
        """
        Custom forward for MCMT-JEPA.
        Returns: (tgt_L0, pred_L0), (tgt_L1, pred_L1), (tgt_L2, pred_L2),
                 (z_view1, z_view2)  for NT-Xent
        """
        device = data.y.device
        subgraph_x_L0, raw_patch_pes, _ = self._gnn_forward(data)

        # Hierarchical pooling
        subgraph_x_L1    = scatter(subgraph_x_L0, data.fine_to_medium, dim=0, reduce='mean')
        raw_patch_pes_L1 = scatter(raw_patch_pes,  data.fine_to_medium, dim=0, reduce='mean')
        subgraph_x_L2    = scatter(subgraph_x_L1, data.medium_to_coarse, dim=0, reduce='mean')
        raw_patch_pes_L2 = scatter(raw_patch_pes_L1, data.medium_to_coarse, dim=0, reduce='mean')

        B = len(data.call_n_patches)

        def _bi(n_patches_list):
            bi = torch.tensor(np.cumsum(n_patches_list))
            return torch.hstack((torch.tensor(0), bi[:-1])).to(device)

        bi_L0 = _bi(data.call_n_patches)
        bi_L1 = _bi(data.n_patches_L1)
        bi_L2 = _bi(data.n_patches_L2)

        # ── L0 JEPA ──────────────────────────────────────────────────────
        # Use 2 context patches (indices: context_subgraph_idx, context_subgraph_idx+1)
        ctx_idx_L0 = data.context_subgraph_idx + bi_L0
        tgt_idx_L0 = torch.vstack([torch.tensor(dt) for dt in data.target_subgraph_idxs]).to(device)
        tgt_idx_L0 += bi_L0.unsqueeze(1)

        # Primary context
        ctx_patch_L0  = subgraph_x_L0[ctx_idx_L0]   # [B, nhid]
        ctx_pe_L0     = raw_patch_pes[ctx_idx_L0]
        ctx_patch_L0  = ctx_patch_L0 + self.patch_rw_encoder(ctx_pe_L0)

        # Secondary context: shift index by +1 (mod n_patches), use 2nd context
        n_patches_per_graph = torch.tensor([c[0] for c in data.call_n_patches], device=device)
        ctx2_offset = (data.context_subgraph_idx + 1) % n_patches_per_graph
        ctx2_idx_L0 = ctx2_offset + bi_L0
        ctx2_patch_L0 = subgraph_x_L0[ctx2_idx_L0]
        ctx2_pe_L0    = raw_patch_pes[ctx2_idx_L0]
        ctx2_patch_L0 = ctx2_patch_L0 + self.patch_rw_encoder(ctx2_pe_L0)

        # Stack 2 contexts: [B, 2, nhid]
        ctx_stack = torch.stack([ctx_patch_L0, ctx2_patch_L0], dim=1)

        # Target embeddings + PEs
        tgt_patch_L0 = subgraph_x_L0[tgt_idx_L0.flatten()]  # [B*nT, nhid]
        tgt_pe_L0    = raw_patch_pes[tgt_idx_L0.flatten()]
        encoded_tgt_pe_L0 = self.patch_rw_encoder(tgt_pe_L0)  # [B*nT, nhid]

        # Target encoder (no grad)
        tgt_x_L0 = tgt_patch_L0.reshape(B, self.num_target_patches, self.nhid)
        with torch.no_grad():
            if hasattr(data, 'coarsen_adj'):
                tgt_rel_L0 = torch.vstack([torch.tensor(dt) for dt in data.target_subgraph_idxs])
                patch_adj_L0 = data.coarsen_adj[
                    torch.arange(B).unsqueeze(1).unsqueeze(2),
                    tgt_rel_L0.unsqueeze(1), tgt_rel_L0.unsqueeze(2)]
                tgt_x_L0 = self.target_encoder_L0(tgt_x_L0, patch_adj_L0, None)
            else:
                tgt_x_L0 = self.target_encoder_L0(tgt_x_L0, None, None)

        # Multi-Context Cross-Attention prediction
        tgt_pe_reshape = encoded_tgt_pe_L0.reshape(B, self.num_target_patches, self.nhid)  # [B, nT, nhid]
        pred_ctx = self.mc_agg_L0(ctx_stack, tgt_pe_reshape)                               # [B, nT, nhid]
        pred_L0  = self.predictor_L0_to_L0(pred_ctx)

        # NT-Xent views: use ctx1 and ctx2 body embeddings
        ctx_x_enc = self.context_encoder_L0(ctx_patch_L0.unsqueeze(1), None, None)  # [B, 1, nhid]
        ctx2_x_enc= self.context_encoder_L0(ctx2_patch_L0.unsqueeze(1), None, None)
        z_view1 = self.proj_head(ctx_x_enc.squeeze(1))    # [B, 128]
        z_view2 = self.proj_head(ctx2_x_enc.squeeze(1))   # [B, 128]

        # ── L0→L1 CROSS-SCALE ────────────────────────────────────────────
        tgt_idx_L1 = torch.vstack([torch.tensor(dt) for dt in data.target_subgraph_idxs_L1]).to(device)
        tgt_idx_L1 += bi_L1.unsqueeze(1)
        nT_L1 = tgt_idx_L1.shape[1]
        tgt_patch_L1 = subgraph_x_L1[tgt_idx_L1.flatten()]
        tgt_pe_L1    = raw_patch_pes_L1[tgt_idx_L1.flatten()]
        encoded_tgt_pe_L1 = self.patch_rw_encoder_L1(tgt_pe_L1)
        tgt_x_L1 = tgt_patch_L1.reshape(B, nT_L1, self.nhid)
        with torch.no_grad():
            tgt_rel_L1 = torch.vstack([torch.tensor(dt) for dt in data.target_subgraph_idxs_L1])
            patch_adj_L1 = data.coarsen_adj_L1[
                torch.arange(B).unsqueeze(1).unsqueeze(2),
                tgt_rel_L1.unsqueeze(1), tgt_rel_L1.unsqueeze(2)]
            tgt_x_L1 = self.target_encoder_L1(tgt_x_L1, patch_adj_L1, None)

        tgt_pe_L1_r = encoded_tgt_pe_L1.reshape(B, nT_L1, self.nhid)
        pred_L1 = self.predictor_L0_to_L1(
            self.mc_agg_L1(ctx_stack, tgt_pe_L1_r))

        # ── L1→L2 (standard, single context) ──────────────────────────────
        ctx_idx_L1 = data.context_subgraph_idx_L1 + bi_L1
        ctx_patch_L1 = subgraph_x_L1[ctx_idx_L1]
        ctx_pe_L1    = raw_patch_pes_L1[ctx_idx_L1]
        ctx_patch_L1 = ctx_patch_L1 + self.patch_rw_encoder_L1(ctx_pe_L1)
        ctx_x_L1 = ctx_patch_L1.unsqueeze(1)
        ctx_mask_L1 = data.mask_L1.flatten()[ctx_idx_L1].reshape(B, 1)
        ctx_x_L1 = self.context_encoder_L1(ctx_x_L1, None, ~ctx_mask_L1)

        tgt_idx_L2 = torch.vstack([torch.tensor(dt) for dt in data.target_subgraph_idxs_L2]).to(device)
        tgt_idx_L2 += bi_L2.unsqueeze(1); nT_L2 = tgt_idx_L2.shape[1]
        tgt_patch_L2 = subgraph_x_L2[tgt_idx_L2.flatten()]
        tgt_pe_L2    = raw_patch_pes_L2[tgt_idx_L2.flatten()]
        encoded_tgt_pe_L2 = self.patch_rw_encoder_L2(tgt_pe_L2)
        tgt_x_L2 = tgt_patch_L2.reshape(B, nT_L2, self.nhid)
        with torch.no_grad():
            tgt_rel_L2 = torch.vstack([torch.tensor(dt) for dt in data.target_subgraph_idxs_L2])
            patch_adj_L2 = data.coarsen_adj_L2[
                torch.arange(B).unsqueeze(1).unsqueeze(2),
                tgt_rel_L2.unsqueeze(1), tgt_rel_L2.unsqueeze(2)]
            tgt_x_L2 = self.target_encoder_L2(tgt_x_L2, patch_adj_L2, None)
        pred_L2 = self.predictor_L1_to_L2(
            ctx_x_L1 + encoded_tgt_pe_L2.reshape(B, nT_L2, self.nhid))

        return ((tgt_x_L0, pred_L0), (tgt_x_L1, pred_L1), (tgt_x_L2, pred_L2),
                (z_view1, z_view2))


# ─────────────────────────────────────────────────────────────
# LOSS
# ─────────────────────────────────────────────────────────────
def _mcmt_loss(model, data):
    """MCMT-JEPA loss: SmoothL1 at 3 scales + NT-Xent auxiliary."""
    if not isinstance(model, GraphHMSJepaMCMT):
        from train.zinc import _compute_loss
        return _compute_loss(model, data, nn.SmoothL1Loss(beta=0.5), 0)

    criterion = nn.SmoothL1Loss(beta=0.5)
    (tx0, ty0), (tx1, ty1), (tx2, ty2), (z1, z2) = model.forward_mcmt(data)

    l0 = criterion(ty0, tx0); l1 = criterion(ty1, tx1); l2 = criterion(ty2, tx2)
    w = model.loss_weights
    loss = w[0]*l0 + w[1]*l1 + w[2]*l2
    # VICReg variance term
    loss = loss + model.var_weight * torch.mean(torch.relu(1.0 - tx0.detach().std(dim=0)))
    # NT-Xent auxiliary
    if len(z1) > 1:
        loss = loss + NT_XENT_WEIGHT * nt_xent_loss(z1, z2)

    return loss, len(ty0)


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


def create_model_mcmt(cfg):
    base = create_model(cfg)
    if not isinstance(base, GraphHMSJepa):
        return base
    node_type, edge_type, nfeat_node, nfeat_edge = _resolve_dataset_types(cfg.dataset)
    model = GraphHMSJepaMCMT(
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
        num_contexts=NUM_CONTEXTS,
    )
    print(f"  [MCMT] num_contexts={NUM_CONTEXTS}  nt_xent_weight={NT_XENT_WEIGHT}")
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
    cfg_path = f"/tmp/exp13_{dataset_name.replace('-','_').lower()}.yaml"
    with open(cfg_path, "w") as f: f.write(DATASET_CONFIGS[dataset_name])
    from core.config import cfg as _cfg
    _cfg.defrost(); _cfg.merge_from_file(cfg_path)
    updated = update_cfg(_cfg, args_str=""); updated.k = 10
    return updated

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
        train_indices, _ = k_fold(dataset, cfg.k); ti = train_indices[0]
        n = min(_PREFLIGHT_MAX_GRAPHS, len(ti))
        subset = dataset[ti[:n]]; subset.transform = transform
        train_list = [x for x in subset]
        bs = min(cfg.train.batch_size, max(1, len(train_list)))
        loader = DataLoader(train_list, batch_size=bs, shuffle=True, num_workers=0, pin_memory=False)
    model = create_model_mcmt(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    data = next(iter(loader)).to(device); optimizer.zero_grad()
    with torch.amp.autocast("cuda", enabled=USE_AMP, dtype=AMP_DTYPE):
        loss, _ = _mcmt_loss(model, data)
    scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()

def run_preflight_all():
    print("\n" + "="*70)
    print("  EXP 13 — PREFLIGHT (Multi-Context JEPA + NT-Xent)")
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
            loss, num_t = _mcmt_loss(model, data)
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
            loss, num_t = _mcmt_loss(model, data)
        step_losses.append(loss.item()); num_targets.append(num_t)
    return None, np.average(step_losses, weights=num_targets)

if __name__ == "__main__":
    skip_pf = os.environ.get("EXP13_SKIP_PREFLIGHT", "0") == "1"
    pf_only = os.environ.get("EXP13_PREFLIGHT_ONLY", "0") == "1"
    print(f"\n  EXP 13: Multi-Context Multi-Target JEPA")
    print(f"  num_contexts={NUM_CONTEXTS} | NT-Xent weight={NT_XENT_WEIGHT} | temp={NT_XENT_TEMPERATURE}")

    if not skip_pf:
        try: run_preflight_all()
        except Exception as e: print(f"\n  PREFLIGHT FAILED: {e!r}"); raise
        if pf_only: sys.exit(0)

    wall_times = {}; tracker_results = {}; total_start = time.time()
    for dataset_name in DATASETS_TO_RUN:
        ds_start = time.time()
        print(f"\n{'#'*70}\n#  EXP 13 — MCMT — DATASET: {dataset_name}\n{'#'*70}")
        updated_cfg = _merge_cfg_for_dataset(dataset_name)
        is_regression = (dataset_name == "ZINC")
        if is_regression:
            tracker_results[dataset_name] = run(updated_cfg, create_dataset, create_model_mcmt, train, test)
        else:
            tracker_results[dataset_name] = run_k_fold(updated_cfg, create_dataset, create_model_mcmt, train, test)
        ds_elapsed = time.time() - ds_start
        wall_times[dataset_name] = ds_elapsed
        print(f"\n  [{dataset_name}] wall time: {ds_elapsed/60:.1f} min")

    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}\n  EXP 13 (MCMT-JEPA) COMPLETE | Total: {total_elapsed/60:.1f} min\n{'='*70}")
    for ds, t in wall_times.items(): print(f"  {ds:<12}  {t/60:>9.1f}m")
    print_exp_tracker_footer(13, "Multi-Context Multi-Target JEPA + NT-Xent", tracker_results)
