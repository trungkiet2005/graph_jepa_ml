# ============================================================
# EXP XX — [Tên experiment]  (Kaggle-ready)
# ============================================================
# Mô tả: [Ý tưởng cải tiến là gì, so với paper gốc cái gì thay đổi]
# Dataset: [MUTAG / PROTEINS / DD / IMDB-B / REDDIT-B / ZINC / ...]
# Baseline: exp_01_hms_jepa_mutag.py
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
# 1. IMPORTS
# ─────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Shared utilities from repo — KHÔNG sửa những file này
from core.config    import cfg, update_cfg
from core.get_data  import create_dataset
from core.get_model import create_model
from core.trainer   import run_k_fold          # hoặc run nếu không dùng k-fold
from core.model     import GraphHMSJepa
from train.zinc     import _compute_loss, _ema_update

# ─────────────────────────────────────────────────────────────
# 2. CONFIG  (copy từ train/configs/<dataset>_hms.yaml rồi chỉnh)
# ─────────────────────────────────────────────────────────────
CONFIG = """
dataset: MUTAG          # <<< đổi dataset ở đây
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

_cfg_path = "/tmp/exp_xx.yaml"
with open(_cfg_path, "w") as f:
    f.write(CONFIG)

# ─────────────────────────────────────────────────────────────
# 3. NEW MODEL  ← PHẦN CHÍNH CẦN THAY ĐỔI CHO MỖI EXPERIMENT
# ─────────────────────────────────────────────────────────────
class MyNewModel(GraphHMSJepa):
    """
    Mô tả thay đổi so với GraphHMSJepa:
    - [thay đổi 1]
    - [thay đổi 2]
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Thêm các layer/module mới ở đây
        # Ví dụ: self.extra_layer = nn.Linear(self.nhid, self.nhid)

    def forward(self, data):
        # Gọi super() nếu chỉ sửa một phần nhỏ
        # hoặc viết lại hoàn toàn nếu thay đổi lớn
        return super().forward(data)

    def encode(self, data):
        # Bắt buộc override nếu forward thay đổi
        return super().encode(data)


def _create_new_model(cfg):
    """Drop-in replacement cho create_model — trả về MyNewModel thay vì model gốc."""
    from core.get_model import create_model as _orig_create_model
    # Lấy kwargs từ cfg như create_model gốc, nhưng dùng class mới
    model = _orig_create_model(cfg)
    # Tạm thời: thay thế class sau khi khởi tạo không dễ;
    # cách sạch hơn là copy hàm create_model và đổi class.
    # TODO: implement properly
    return model


# ─────────────────────────────────────────────────────────────
# 4. TRAIN / TEST  (thường giữ nguyên, chỉ sửa nếu loss thay đổi)
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

    return None, np.average(step_losses, weights=num_targets)


@torch.no_grad()
def test(loader, model, evaluator, device, criterion_type=0):
    criterion = torch.nn.SmoothL1Loss(beta=0.5)
    step_losses, num_targets = [], []
    for data in loader:
        data = data.to(device)
        loss, num_t = _compute_loss(model, data, criterion, criterion_type)
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

    # Dùng create_model gốc hoặc _create_new_model tùy experiment
    run_k_fold(cfg, create_dataset, create_model, train, test)
