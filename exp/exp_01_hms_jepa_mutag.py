# ============================================================
# EXP 01 — HMS-JEPA Baseline on MUTAG  (Kaggle-ready)
# ============================================================
# Reproduce the HMS-JEPA result from experiment 01.
# This file is self-contained: run it directly on Kaggle after
# uploading / pasting — no extra files needed.
#
# Usage on Kaggle:
#   !python exp_01_hms_jepa_mutag.py
#   or just run each cell if you paste sections into a notebook.
# ============================================================

# ─────────────────────────────────────────────────────────────
# 0. ENVIRONMENT SETUP  (run once per Kaggle session)
# ─────────────────────────────────────────────────────────────
import os, sys, subprocess

REPO_URL  = "https://github.com/trungkiet2005/graph_jepa_ml.git"
REPO_DIR  = "/kaggle/working/graph_jepa_ml"

if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)

os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)

# System dependency for METIS partitioning
subprocess.run(["apt-get", "install", "-y", "libmetis-dev"],
               check=True, capture_output=True)
os.environ["METIS_DLL"] = "/usr/lib/x86_64-linux-gnu/libmetis.so"

# Python dependencies (quiet install)
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
# 1. IMPORTS  (shared utilities come from the cloned repo)
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
from train.zinc     import _compute_loss, _ema_update   # shared loss helpers

# ─────────────────────────────────────────────────────────────
# 2. CONFIG  (inline — mirrors train/configs/mutag_hms.yaml)
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

# Write config to a temp file so cfg can load it
_cfg_path = "/tmp/mutag_hms_exp01.yaml"
with open(_cfg_path, "w") as f:
    f.write(CONFIG)

# ─────────────────────────────────────────────────────────────
# 3. MODEL  (HMS-JEPA — no changes in exp_01, pure baseline)
# ─────────────────────────────────────────────────────────────
# NOTE: For future experiments, define your new model class HERE
# and pass it via create_model override below.
#
# Example skeleton for exp_02+:
#
# class MyImprovedModel(GraphHMSJepa):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         # add new components here
#
#     def forward(self, data):
#         # override forward with new method
#         ...
#
#     def encode(self, data):
#         # override encode for inference
#         ...


# ─────────────────────────────────────────────────────────────
# 4. TRAIN / TEST LOOPS  (standard — shared with all exp files)
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

        loss, num_t = _compute_loss(model, data, criterion, criterion_type)
        step_losses.append(loss.item())
        num_targets.append(num_t)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            _ema_update(model, momentum_weight)

    epoch_loss = np.average(step_losses, weights=num_targets)
    return None, epoch_loss


@torch.no_grad()
def test(loader, model, evaluator, device, criterion_type=0):
    criterion = torch.nn.SmoothL1Loss(beta=0.5)
    step_losses, num_targets = [], []
    for data in loader:
        data = data.to(device)
        loss, num_t = _compute_loss(model, data, criterion, criterion_type)
        step_losses.append(loss.item())
        num_targets.append(num_t)

    epoch_loss = np.average(step_losses, weights=num_targets)
    return None, epoch_loss


# ─────────────────────────────────────────────────────────────
# 5. ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cfg.merge_from_file(_cfg_path)
    cfg = update_cfg(cfg, args_str="")
    cfg.k = 10

    run_k_fold(cfg, create_dataset, create_model, train, test)
