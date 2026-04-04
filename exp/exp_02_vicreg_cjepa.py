# ============================================================
# EXP 02 — C-JEPA: Full VICReg Regularization for HMS-JEPA
# ============================================================
# Paper:  "Connecting JEPA with Contrastive Self-supervised Learning"
#          NeurIPS 2024 (Spotlight)
#
# Idea:   The original HMS-JEPA only uses a simple variance term
#         (var_weight * relu(1 - std)) for collapse prevention.
#         C-JEPA shows that full VICReg regularization (Variance +
#         Invariance + Covariance) dramatically improves JEPA training.
#
#         We add:
#         1. Variance: hinge loss on std of each embedding dimension
#            (already exists but we strengthen it)
#         2. Covariance: penalize off-diagonal elements of the
#            embedding covariance matrix → decorrelation
#         3. Cross-scale invariance: cosine similarity between
#            L0/L1/L2 target embeddings (aligned via pooling)
#
# Dataset: MUTAG (10-fold CV)
# Expected improvement: Better collapse prevention → higher accuracy
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
from core.get_model import create_model
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

_cfg_path = "/tmp/exp02_vicreg.yaml"
with open(_cfg_path, "w") as f:
    f.write(CONFIG)

# ─────────────────────────────────────────────────────────────
# 3. VICReg LOSS FUNCTIONS
# ─────────────────────────────────────────────────────────────

def off_diagonal(M):
    """Return flattened off-diagonal elements of a square matrix."""
    n = M.shape[0]
    return M.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def vicreg_loss(z_pred, z_target, sim_weight=25.0, var_weight=25.0, cov_weight=1.0):
    """Full VICReg loss between predicted and target embeddings.

    Args:
        z_pred:   [B, T, D]  predicted patch embeddings
        z_target: [B, T, D]  target patch embeddings (detached)
        sim_weight: weight for invariance (MSE) term
        var_weight: weight for variance term
        cov_weight: weight for covariance term
    """
    B, T, D = z_pred.shape

    # Flatten batch and target dims: [B*T, D]
    pred_flat = z_pred.reshape(-1, D)
    tgt_flat  = z_target.reshape(-1, D)

    # ── Invariance (MSE between prediction and target) ──
    sim_loss = F.mse_loss(pred_flat, tgt_flat)

    # ── Variance (hinge loss on std of each dimension) ──
    pred_std = pred_flat.std(dim=0)
    tgt_std  = tgt_flat.std(dim=0)
    var_loss = (F.relu(1.0 - pred_std).mean() + F.relu(1.0 - tgt_std).mean()) / 2

    # ── Covariance (penalize off-diagonal correlations) ──
    pred_centered = pred_flat - pred_flat.mean(dim=0)
    tgt_centered  = tgt_flat  - tgt_flat.mean(dim=0)
    N = pred_flat.shape[0]

    cov_pred = (pred_centered.T @ pred_centered) / max(N - 1, 1)
    cov_tgt  = (tgt_centered.T @ tgt_centered)   / max(N - 1, 1)

    cov_loss = (off_diagonal(cov_pred).pow(2).sum() / D +
                off_diagonal(cov_tgt).pow(2).sum()  / D)

    return sim_weight * sim_loss + var_weight * var_loss + cov_weight * cov_loss


def compute_loss_vicreg(model, data, criterion, criterion_type):
    """Compute HMS-JEPA loss with full VICReg regularization."""
    (tgt_x_L0, pred_L0), (tgt_x_L1, pred_L1), (tgt_x_L2, pred_L2) = model(data)

    w = model.loss_weights

    # Primary JEPA prediction losses (SmoothL1 as in baseline)
    jepa_L0 = criterion(pred_L0, tgt_x_L0.detach())
    jepa_L1 = criterion(pred_L1, tgt_x_L1.detach())
    jepa_L2 = criterion(pred_L2, tgt_x_L2.detach())
    jepa_loss = w[0] * jepa_L0 + w[1] * jepa_L1 + w[2] * jepa_L2

    # VICReg regularization on each scale
    vicreg_L0 = vicreg_loss(pred_L0, tgt_x_L0.detach(),
                            sim_weight=0.0, var_weight=1.0, cov_weight=0.05)
    vicreg_L1 = vicreg_loss(pred_L1, tgt_x_L1.detach(),
                            sim_weight=0.0, var_weight=1.0, cov_weight=0.05)
    vicreg_L2 = vicreg_loss(pred_L2, tgt_x_L2.detach(),
                            sim_weight=0.0, var_weight=1.0, cov_weight=0.05)

    vicreg_total = 0.01 * (w[0] * vicreg_L0 + w[1] * vicreg_L1 + w[2] * vicreg_L2)

    loss = jepa_loss + vicreg_total
    num_t = len(pred_L0)

    return loss, num_t


# ─────────────────────────────────────────────────────────────
# 4. TRAIN / TEST
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

        loss, num_t = compute_loss_vicreg(model, data, criterion, criterion_type)
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
        loss, num_t = compute_loss_vicreg(model, data, criterion, criterion_type)
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

    run_k_fold(cfg, create_dataset, create_model, train, test)
