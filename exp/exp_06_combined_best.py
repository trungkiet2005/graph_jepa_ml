# ============================================================
# EXP 06 — COMBINED: VICReg + LayerAttn + Adaptive Weights
# ============================================================
# This experiment combines the THREE best ideas from exp_02-05
# into a single unified model, targeting maximum performance:
#
# 1. Full VICReg regularization (from C-JEPA, NeurIPS 2024)
#    → Prevents representation collapse with variance + covariance
#
# 2. Layer-wise attention pooling (from HISTOGRAPH, 2026)
#    → Uses ALL GNN layer outputs, not just the last one
#
# 3. Adaptive loss weighting (from Kendall et al., CVPR 2018)
#    → Learns optimal scale weights instead of fixed [1,0.5,0.25]
#
# This is the "kitchen sink" experiment — if individual ideas
# each improve by X%, the combination should push further.
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

_cfg_path = "/tmp/exp06_combined.yaml"
with open(_cfg_path, "w") as f:
    f.write(CONFIG)

# ─────────────────────────────────────────────────────────────
# 3. BUILDING BLOCKS
# ─────────────────────────────────────────────────────────────

class LayerAttentionPool(nn.Module):
    """Learnable attention-weighted pooling across GNN layers."""

    def __init__(self, nhid, nlayers):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(nhid, nhid // 4),
            nn.ReLU(),
            nn.Linear(nhid // 4, 1),
        )

    def forward(self, layer_outputs):
        stacked = torch.stack(layer_outputs, dim=1)  # [N, L, D]
        scores = self.attn(stacked)                    # [N, L, 1]
        weights = F.softmax(scores, dim=1)
        return (weights * stacked).sum(dim=1)          # [N, D]


def off_diagonal(M):
    n = M.shape[0]
    return M.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def vicreg_regularization(z, var_weight=1.0, cov_weight=0.05):
    """VICReg variance + covariance regularization (no invariance term).

    Args:
        z: [B, T, D] or [B*T, D] — embeddings to regularize
    Returns:
        scalar loss
    """
    if z.dim() == 3:
        z = z.reshape(-1, z.shape[-1])
    D = z.shape[-1]

    # Variance hinge
    std = z.std(dim=0)
    var_loss = F.relu(1.0 - std).mean()

    # Covariance decorrelation
    z_centered = z - z.mean(dim=0)
    cov = (z_centered.T @ z_centered) / max(z.shape[0] - 1, 1)
    cov_loss = off_diagonal(cov).pow(2).sum() / D

    return var_weight * var_loss + cov_weight * cov_loss


# ─────────────────────────────────────────────────────────────
# 4. COMBINED MODEL
# ─────────────────────────────────────────────────────────────

class GraphHMSJepaCombined(GraphHMSJepa):
    """HMS-JEPA with:
    1. Layer-wise attention pooling (richer GNN features)
    2. Adaptive loss weighting (learned uncertainty)
    VICReg regularization is applied externally in the loss function.
    """

    def __init__(self, *args, nlayer_gnn=2, **kwargs):
        super().__init__(*args, nlayer_gnn=nlayer_gnn, **kwargs)

        # [Component 1] Layer attention pooling
        self.layer_attn = LayerAttentionPool(self.nhid, nlayer_gnn)

        # [Component 2] Adaptive loss weights (homoscedastic uncertainty)
        self.log_var_L0 = nn.Parameter(torch.tensor(-0.69))
        self.log_var_L1 = nn.Parameter(torch.tensor(0.0))
        self.log_var_L2 = nn.Parameter(torch.tensor(0.69))

    def get_adaptive_weights(self):
        precision_L0 = torch.exp(-self.log_var_L0)
        precision_L1 = torch.exp(-self.log_var_L1)
        precision_L2 = torch.exp(-self.log_var_L2)
        reg = 0.5 * (self.log_var_L0 + self.log_var_L1 + self.log_var_L2)
        return precision_L0, precision_L1, precision_L2, reg

    def _gnn_forward(self, data):
        """Override: layer-wise attention pooling across all GNN outputs."""
        x = self.input_encoder(data.x).squeeze()

        edge_attr = data.edge_attr
        if edge_attr is None:
            edge_attr = data.edge_index.new_zeros(data.edge_index.size(-1)).float().unsqueeze(-1)
        edge_attr = self.edge_encoder(edge_attr)

        x        = x[data.subgraphs_nodes_mapper]
        edge_index = data.combined_subgraphs
        e        = edge_attr[data.subgraphs_edges_mapper]
        batch_x  = data.subgraphs_batch
        pes      = data.rw_pos_enc[data.subgraphs_nodes_mapper]
        raw_patch_pes = scatter(pes, batch_x, dim=0, reduce='max')

        layer_outputs = []
        for i, gnn in enumerate(self.gnns):
            if i > 0:
                subgraph = scatter(x, batch_x, dim=0, reduce=self.pooling)[batch_x]
                subgraph = scatter(x, batch_x, dim=0, reduce=self.pooling)[batch_x]
                x = x + self.U[i - 1](subgraph)
                x = scatter(x, data.subgraphs_nodes_mapper, dim=0, reduce='mean')[data.subgraphs_nodes_mapper]
            x = gnn(x, edge_index, e)
            layer_pool = scatter(x, batch_x, dim=0, reduce=self.pooling)
            layer_outputs.append(layer_pool)

        # Attention-weighted combination across GNN layers
        subgraph_x_L0 = self.layer_attn(layer_outputs)
        return subgraph_x_L0, raw_patch_pes, batch_x


def create_model_combined(cfg):
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

    return GraphHMSJepaCombined(
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
    )


# ─────────────────────────────────────────────────────────────
# 5. COMBINED LOSS FUNCTION
# ─────────────────────────────────────────────────────────────

def compute_loss_combined(model, data, criterion, criterion_type):
    """Combined loss: adaptive weights + VICReg regularization."""
    (tgt_x_L0, pred_L0), (tgt_x_L1, pred_L1), (tgt_x_L2, pred_L2) = model(data)

    # Per-scale prediction losses
    if criterion_type == 0:
        l0 = criterion(pred_L0, tgt_x_L0.detach())
        l1 = criterion(pred_L1, tgt_x_L1.detach())
        l2 = criterion(pred_L2, tgt_x_L2.detach())
    else:
        l0 = F.mse_loss(pred_L0, tgt_x_L0.detach())
        l1 = F.mse_loss(pred_L1, tgt_x_L1.detach())
        l2 = F.mse_loss(pred_L2, tgt_x_L2.detach())

    # [Adaptive weighting] Homoscedastic uncertainty
    w0, w1, w2, reg = model.get_adaptive_weights()
    jepa_loss = 0.5 * (w0 * l0 + w1 * l1 + w2 * l2) + reg

    # [VICReg] Variance + Covariance regularization on predictions
    vicreg_L0 = vicreg_regularization(pred_L0, var_weight=1.0, cov_weight=0.05)
    vicreg_L1 = vicreg_regularization(pred_L1, var_weight=1.0, cov_weight=0.05)
    vicreg_L2 = vicreg_regularization(pred_L2, var_weight=1.0, cov_weight=0.05)
    vicreg_total = 0.01 * (vicreg_L0 + vicreg_L1 + vicreg_L2)

    loss = jepa_loss + vicreg_total
    return loss, len(pred_L0)


# ─────────────────────────────────────────────────────────────
# 6. TRAIN / TEST
# ─────────────────────────────────────────────────────────────

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

        loss, num_t = compute_loss_combined(model, data, criterion, criterion_type)
        step_losses.append(loss.item())
        num_targets.append(num_t)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            _ema_update(model, momentum_weight)

    with torch.no_grad():
        w0, w1, w2, _ = model.get_adaptive_weights()
        print(f"  [Weights] L0={w0.item():.3f} L1={w1.item():.3f} L2={w2.item():.3f}")

    return None, np.average(step_losses, weights=num_targets)


@torch.no_grad()
def test(loader, model, evaluator, device, criterion_type=0):
    criterion = torch.nn.SmoothL1Loss(beta=0.5)
    step_losses, num_targets = [], []
    for data in loader:
        data = data.to(device)
        loss, num_t = compute_loss_combined(model, data, criterion, criterion_type)
        step_losses.append(loss.item())
        num_targets.append(num_t)

    return None, np.average(step_losses, weights=num_targets)


# ─────────────────────────────────────────────────────────────
# 7. ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cfg.merge_from_file(_cfg_path)
    cfg = update_cfg(cfg, args_str="")
    cfg.k = 10

    run_k_fold(cfg, create_dataset, create_model_combined, train, test)
