# ============================================================
# EXP 04 — Multi-View Re-Masking + Stochastic Context Augmentation
# ============================================================
# Papers: "GraphMAE2: Decoding-Enhanced Masked Graph Learner" (WWW 2023)
#         "V-JEPA 2: Self-Supervised Video Foundation Models" (Meta 2025)
#
# Idea:   The baseline forward pass uses ONE context embedding to
#         predict all targets. This creates a single-view bottleneck.
#
#         GraphMAE2 showed that re-masking the encoded representation
#         multiple times and averaging predictions acts as a powerful
#         stochastic regularizer — like multi-crop in DINO.
#
#         V-JEPA 2 showed that high masking ratios (90%+) with multiple
#         context views improve representation quality.
#
#         We combine both ideas:
#         1. Generate K random dropout masks on the context embedding
#         2. Predict targets from each masked context independently
#         3. Average the predictions → smoother, more robust training
#
# Key changes:
#   - Override forward() to apply K dropout views on context
#   - Average predictions across views
#   - No architecture change — same model, better training signal
#
# Dataset: MUTAG (10-fold CV)
# ============================================================

# ─────────────────────────────────────────────────────────────
# 0. ENVIRONMENT SETUP
# ─────────────────────────────────────────────────────────────
import os, sys, subprocess

REPO_URL = "https://github.com/trungkiet2005/graph_jepa_ml.git"
REPO_DIR = "/kaggle/working/graph_jepa_ml"

if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)

os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)

subprocess.run(["apt-get", "install", "-y", "libmetis-dev"],
               check=True, capture_output=True)
os.environ["METIS_DLL"] = "/usr/lib/x86_64-linux-gnu/libmetis.so"

import torch as _torch
torch_ver = _torch.__version__
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

from core.config    import cfg, update_cfg
from core.get_data  import create_dataset
from core.trainer   import run_k_fold
from core.model     import GraphHMSJepa
from core.model_utils.elements import MLP
from train.zinc     import _ema_update

# ─────────────────────────────────────────────────────────────
# 2. CONFIG
# ─────────────────────────────────────────────────────────────
CONFIG = """
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
  batch_size: 128
  lr: 0.0005
  runs: 5
metis:
  n_patches: 32
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
"""

_cfg_path = "/tmp/exp04_multiview.yaml"
with open(_cfg_path, "w") as f:
    f.write(CONFIG)

# ─────────────────────────────────────────────────────────────
# 3. MODEL — Multi-View Re-Masking HMS-JEPA
# ─────────────────────────────────────────────────────────────

# Hyperparameters for multi-view
NUM_VIEWS = 4         # Number of context dropout views
VIEW_DROP_RATE = 0.15  # Feature dropout rate per view


class GraphHMSJepaMultiView(GraphHMSJepa):
    """HMS-JEPA with multi-view re-masking of context embeddings.

    During training, applies K random feature dropout masks to the
    context embedding and averages the predictions. During eval,
    uses the standard single-view forward.
    """

    def __init__(self, *args, num_views=NUM_VIEWS, view_drop_rate=VIEW_DROP_RATE, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_views = num_views
        self.view_drop_rate = view_drop_rate

    def forward(self, data):
        """Multi-view forward: average predictions over K context dropout views."""
        device = data.y.device
        subgraph_x_L0, raw_patch_pes, _ = self._gnn_forward(data)

        # Hierarchical pooling (same as base)
        subgraph_x_L1    = scatter(subgraph_x_L0, data.fine_to_medium,   dim=0, reduce='mean')
        raw_patch_pes_L1 = scatter(raw_patch_pes,  data.fine_to_medium,   dim=0, reduce='mean')
        subgraph_x_L2    = scatter(subgraph_x_L1,  data.medium_to_coarse, dim=0, reduce='mean')
        raw_patch_pes_L2 = scatter(raw_patch_pes_L1, data.medium_to_coarse, dim=0, reduce='mean')

        B = len(data.call_n_patches)

        def _batch_indexer(n_patches_list):
            bi = torch.tensor(np.cumsum(n_patches_list))
            return torch.hstack((torch.tensor(0), bi[:-1])).to(device)

        bi_L0 = _batch_indexer(data.call_n_patches)
        bi_L1 = _batch_indexer(data.n_patches_L1)
        bi_L2 = _batch_indexer(data.n_patches_L2)

        # ── L0 context/target indices ──
        ctx_idx_L0 = data.context_subgraph_idx + bi_L0
        tgt_idx_L0 = torch.vstack([torch.tensor(dt) for dt in data.target_subgraph_idxs]).to(device)
        tgt_idx_L0 += bi_L0.unsqueeze(1)

        ctx_patch_L0 = subgraph_x_L0[ctx_idx_L0]
        tgt_patch_L0 = subgraph_x_L0[tgt_idx_L0.flatten()]
        ctx_pe_L0    = raw_patch_pes[ctx_idx_L0]
        tgt_pe_L0    = raw_patch_pes[tgt_idx_L0.flatten()]

        ctx_patch_L0      = ctx_patch_L0 + self.patch_rw_encoder(ctx_pe_L0)
        encoded_tgt_pe_L0 = self.patch_rw_encoder(tgt_pe_L0)

        tgt_x_L0 = tgt_patch_L0.reshape(B, self.num_target_patches, self.nhid)

        # ── Target encodings (no grad, same for all views) ──
        with torch.no_grad():
            if hasattr(data, 'coarsen_adj'):
                tgt_rel_L0 = torch.vstack([torch.tensor(dt) for dt in data.target_subgraph_idxs])
                patch_adj_L0 = data.coarsen_adj[
                    torch.arange(B).unsqueeze(1).unsqueeze(2),
                    tgt_rel_L0.unsqueeze(1), tgt_rel_L0.unsqueeze(2)]
                tgt_x_L0 = self.target_encoder_L0(tgt_x_L0, patch_adj_L0, None)
            else:
                tgt_x_L0 = self.target_encoder_L0(tgt_x_L0, None, None)

        # ── L1 targets ──
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

        # ── L2 targets ──
        ctx_idx_L1 = data.context_subgraph_idx_L1 + bi_L1
        ctx_patch_L1 = subgraph_x_L1[ctx_idx_L1]
        ctx_pe_L1    = raw_patch_pes_L1[ctx_idx_L1]
        ctx_patch_L1 = ctx_patch_L1 + self.patch_rw_encoder_L1(ctx_pe_L1)

        tgt_idx_L2 = torch.vstack([torch.tensor(dt) for dt in data.target_subgraph_idxs_L2]).to(device)
        tgt_idx_L2 += bi_L2.unsqueeze(1)
        nT_L2 = tgt_idx_L2.shape[1]
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

        # ══════════════════════════════════════════════════════
        # Multi-view: K dropout views on context, average preds
        # ══════════════════════════════════════════════════════
        K = self.num_views if self.training else 1
        preds_L0_list, preds_L1_list, preds_L2_list = [], [], []

        for _ in range(K):
            # Apply feature dropout to context embedding
            if self.training and K > 1:
                ctx_drop = F.dropout(ctx_patch_L0, p=self.view_drop_rate, training=True)
            else:
                ctx_drop = ctx_patch_L0

            ctx_x_L0 = ctx_drop.unsqueeze(1)
            ctx_mask_L0 = data.mask.flatten()[ctx_idx_L0].reshape(B, self.num_context_patches)
            ctx_x_L0 = self.context_encoder_L0(
                ctx_x_L0,
                data.coarsen_adj if hasattr(data, 'coarsen_adj') else None,
                ~ctx_mask_L0)

            # L0 prediction
            pred_L0 = self.predictor_L0_to_L0(
                ctx_x_L0 + encoded_tgt_pe_L0.reshape(B, self.num_target_patches, self.nhid))
            preds_L0_list.append(pred_L0)

            # L1 prediction (using same dropped context)
            pred_L1 = self.predictor_L0_to_L1(
                ctx_x_L0 + encoded_tgt_pe_L1.reshape(B, nT_L1, self.nhid))
            preds_L1_list.append(pred_L1)

            # L2 prediction (from L1 context with dropout)
            if self.training and K > 1:
                ctx_L1_drop = F.dropout(ctx_patch_L1, p=self.view_drop_rate, training=True)
            else:
                ctx_L1_drop = ctx_patch_L1

            ctx_x_L1 = ctx_L1_drop.unsqueeze(1)
            ctx_mask_L1 = data.mask_L1.flatten()[ctx_idx_L1].reshape(B, 1)
            ctx_x_L1 = self.context_encoder_L1(ctx_x_L1, None, ~ctx_mask_L1)

            pred_L2 = self.predictor_L1_to_L2(
                ctx_x_L1 + encoded_tgt_pe_L2.reshape(B, nT_L2, self.nhid))
            preds_L2_list.append(pred_L2)

        # Average across views
        pred_L0 = torch.stack(preds_L0_list).mean(0)
        pred_L1 = torch.stack(preds_L1_list).mean(0)
        pred_L2 = torch.stack(preds_L2_list).mean(0)

        return (tgt_x_L0, pred_L0), (tgt_x_L1, pred_L1), (tgt_x_L2, pred_L2)


def create_model_multiview(cfg):
    """Create GraphHMSJepaMultiView."""
    ds = cfg.dataset
    feat_map = {
        'MUTAG':          (7, 4, 'Linear', 'Linear', 2),
        'PROTEINS':       (3, 1, 'Linear', 'Linear', 2),
        'DD':             (89, 1, 'Linear', 'Linear', 2),
        'REDDIT-BINARY':  (1, 1, 'Linear', 'Linear', 2),
        'REDDIT-MULTI-5K':(1, 1, 'Linear', 'Linear', 5),
        'IMDB-BINARY':    (1, 1, 'Linear', 'Linear', 2),
        'IMDB-MULTI':     (1, 1, 'Linear', 'Linear', 3),
        'ZINC':           (28, 4, 'Discrete', 'Discrete', 1),
    }
    nfeat_node, nfeat_edge, node_type, edge_type, nout = feat_map[ds]

    return GraphHMSJepaMultiView(
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
        num_target_patches_L1=cfg.jepa.num_targets_L1,
        num_target_patches_L2=cfg.jepa.num_targets_L2,
        loss_weights=cfg.jepa.loss_weights,
        var_weight=cfg.jepa.var_weight,
        num_views=NUM_VIEWS,
        view_drop_rate=VIEW_DROP_RATE,
    )


# ─────────────────────────────────────────────────────────────
# 4. TRAIN / TEST
# ─────────────────────────────────────────────────────────────

def _compute_loss_hms(model, data, criterion, criterion_type):
    """Standard HMS-JEPA loss (works for any HMS model)."""
    (tx0, ty0), (tx1, ty1), (tx2, ty2) = model(data)
    if criterion_type == 0:
        l0 = criterion(tx0, ty0)
        l1 = criterion(tx1, ty1)
        l2 = criterion(tx2, ty2)
    else:
        l0 = F.mse_loss(tx0, ty0)
        l1 = F.mse_loss(tx1, ty1)
        l2 = F.mse_loss(tx2, ty2)
    w = model.loss_weights
    loss = w[0] * l0 + w[1] * l1 + w[2] * l2
    loss = loss + model.var_weight * torch.mean(torch.relu(1.0 - tx0.detach().std(dim=0)))
    return loss, len(ty0)


def train(train_loader, model, optimizer, evaluator, device,
          momentum_weight, sharp=None, criterion_type=0):
    criterion = torch.nn.SmoothL1Loss(beta=0.5)
    step_losses, num_targets = [], []
    for data in train_loader:
        if model.use_lap:
            batch_pos_enc = data.lap_pos_enc
            sign_flip = torch.rand(batch_pos_enc.size(1))
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            data.lap_pos_enc = batch_pos_enc * sign_flip.unsqueeze(0)
        data = data.to(device)
        optimizer.zero_grad()

        loss, num_t = _compute_loss_hms(model, data, criterion, criterion_type)
        step_losses.append(loss.item())
        num_targets.append(num_t)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            _ema_update(model, momentum_weight)

    return None, np.average(step_losses, weights=num_targets)


@torch.no_grad()
def test(loader, model, evaluator, device, criterion_type=0):
    criterion = torch.nn.SmoothL1Loss(beta=0.5)
    step_losses, num_targets = [], []
    for data in loader:
        data = data.to(device)
        loss, num_t = _compute_loss_hms(model, data, criterion, criterion_type)
        step_losses.append(loss.item())
        num_targets.append(num_t)

    return None, np.average(step_losses, weights=num_targets)


# ─────────────────────────────────────────────────────────────
# 5. ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cfg.merge_from_file(_cfg_path)
    cfg = update_cfg(cfg, args_str="")
    cfg.k = 10

    run_k_fold(cfg, create_dataset, create_model_multiview, train, test)
