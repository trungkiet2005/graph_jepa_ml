# ============================================================
# EXP 05 — Adaptive Loss Weighting via Uncertainty (Multi-Task)
# ============================================================
# Papers: "Multi-Task Learning Using Uncertainty to Weigh Losses"
#          (Kendall et al., CVPR 2018)
#         "Exploring Correlations of Self-Supervised Tasks for Graphs"
#          (GraphTCM, ICML 2024)
#
# Idea:   HMS-JEPA uses FIXED loss weights [1.0, 0.5, 0.25] for
#         the three scale levels. But different datasets need
#         different balancing — MUTAG (small molecules) may benefit
#         more from fine-scale (L0), while DD (larger proteins) may
#         benefit from coarse-scale (L2).
#
#         Kendall et al. showed that homoscedastic uncertainty can
#         automatically learn optimal task weights:
#           L = (1/2σ²_i) * L_i + log(σ_i)
#
#         We learn σ_0, σ_1, σ_2 as trainable parameters, so the
#         model automatically finds the best weighting per dataset.
#
# Key changes:
#   - Add 3 learnable log-variance parameters (log_σ²)
#   - Replace fixed loss_weights with uncertainty-based weighting
#   - No architecture change — same HMS-JEPA, better optimization
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

from core.config    import cfg, update_cfg
from core.get_data  import create_dataset
from core.trainer   import run_k_fold
from core.model     import GraphHMSJepa
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

_cfg_path = "/tmp/exp05_adaptive.yaml"
with open(_cfg_path, "w") as f:
    f.write(CONFIG)

# ─────────────────────────────────────────────────────────────
# 3. MODEL — HMS-JEPA with Learnable Uncertainty Weights
# ─────────────────────────────────────────────────────────────

class GraphHMSJepaAdaptive(GraphHMSJepa):
    """HMS-JEPA with homoscedastic uncertainty-based loss weighting.

    Instead of fixed loss_weights=[1.0, 0.5, 0.25], we learn
    log(σ²) for each scale and use:
      L_total = Σ (1/(2σ²_i)) * L_i + log(σ_i)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Learnable log-variance for each scale (initialized near baseline weights)
        # log(1/(2*w)) → σ² = 1/(2*w), so:
        #   w=1.0 → log_var = log(0.5) ≈ -0.69
        #   w=0.5 → log_var = log(1.0) = 0.0
        #   w=0.25 → log_var = log(2.0) ≈ 0.69
        self.log_var_L0 = nn.Parameter(torch.tensor(-0.69))
        self.log_var_L1 = nn.Parameter(torch.tensor(0.0))
        self.log_var_L2 = nn.Parameter(torch.tensor(0.69))

    def get_adaptive_weights(self):
        """Compute adaptive weights from learned log-variances.

        Returns: (w0, w1, w2, regularization_term)
        """
        precision_L0 = torch.exp(-self.log_var_L0)
        precision_L1 = torch.exp(-self.log_var_L1)
        precision_L2 = torch.exp(-self.log_var_L2)
        reg = 0.5 * (self.log_var_L0 + self.log_var_L1 + self.log_var_L2)
        return precision_L0, precision_L1, precision_L2, reg


def create_model_adaptive(cfg):
    """Create GraphHMSJepaAdaptive."""
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

    return GraphHMSJepaAdaptive(
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
# 4. LOSS + TRAIN / TEST with adaptive weighting
# ─────────────────────────────────────────────────────────────

def compute_loss_adaptive(model, data, criterion, criterion_type):
    """HMS-JEPA loss with learned uncertainty weights."""
    (tgt_x_L0, pred_L0), (tgt_x_L1, pred_L1), (tgt_x_L2, pred_L2) = model(data)

    if criterion_type == 0:
        l0 = criterion(pred_L0, tgt_x_L0.detach())
        l1 = criterion(pred_L1, tgt_x_L1.detach())
        l2 = criterion(pred_L2, tgt_x_L2.detach())
    else:
        l0 = F.mse_loss(pred_L0, tgt_x_L0.detach())
        l1 = F.mse_loss(pred_L1, tgt_x_L1.detach())
        l2 = F.mse_loss(pred_L2, tgt_x_L2.detach())

    # Adaptive weights via homoscedastic uncertainty
    w0, w1, w2, reg = model.get_adaptive_weights()
    loss = 0.5 * (w0 * l0 + w1 * l1 + w2 * l2) + reg

    # Variance regularization (same as baseline)
    loss = loss + model.var_weight * torch.mean(torch.relu(1.0 - tgt_x_L0.detach().std(dim=0)))

    return loss, len(pred_L0)


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

        loss, num_t = compute_loss_adaptive(model, data, criterion, criterion_type)
        step_losses.append(loss.item())
        num_targets.append(num_t)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            _ema_update(model, momentum_weight)

    # Log learned weights (useful for analysis)
    with torch.no_grad():
        w0, w1, w2, _ = model.get_adaptive_weights()
        print(f"  [Adaptive weights] L0={w0.item():.3f}, L1={w1.item():.3f}, L2={w2.item():.3f}")

    return None, np.average(step_losses, weights=num_targets)


@torch.no_grad()
def test(loader, model, evaluator, device, criterion_type=0):
    criterion = torch.nn.SmoothL1Loss(beta=0.5)
    step_losses, num_targets = [], []
    for data in loader:
        data = data.to(device)
        loss, num_t = compute_loss_adaptive(model, data, criterion, criterion_type)
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

    run_k_fold(cfg, create_dataset, create_model_adaptive, train, test)
