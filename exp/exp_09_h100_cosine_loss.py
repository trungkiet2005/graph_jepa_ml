# ============================================================
# EXP 09 — HMS-JEPA + Cosine Embedding Loss + Improved EMA Schedule
# ============================================================
# Novelty: Replace SmoothL1 latent-prediction loss with a Cosine
# Embedding Loss + Variance-Covariance Regularisation (VICReg-lite),
# combined with a cosine-annealing EMA momentum schedule.
#
# Motivation (from tracker): SmoothL1 (L1-based regression) is sensitive
# to scale and may encourage collapsed representations. Cosine similarity
# is scale-invariant and better reflects directional alignment in the
# latent space (exactly what JEPA needs: "predict *direction* of target").
#
# Theoretical grounding:
#   - Chen & He, "Exploring Simple Siamese Representation Learning",
#     CVPR 2021 — cosine similarity as the collapse-free SSL objective.
#   - Bardes et al., "VICReg: Variance-Invariance-Covariance
#     Regularization for Self-Supervised Learning", ICLR 2022.
#   - Assran et al., "Self-Supervised Learning from Images with a
#     Joint-Embedding Predictive Architecture (I-JEPA)", CVPR 2023 —
#     cosine prediction + EMA momentum cosine schedule.
#   - Grill et al., "Bootstrap Your Own Latent (BYOL)", NeurIPS 2020 —
#     EMA momentum warm-up schedule proven critical.
#   - Liu et al., "Graph Self-supervised Learning with Accurate
#     Discrepancy Learning", NeurIPS 2022.
#
# Key changes vs EXP 07 baseline:
#   1. Loss = negative cosine similarity (per-target-patch mean), NOT SmoothL1.
#   2. VICReg variance term weight is *annealed* from 0.1 → 0 over training
#      (avoids the 4M scale blow-up seen in exp02).
#   3. EMA momentum follows cosine schedule: start_m=0.996, end_m=1.0
#      (same as I-JEPA / BYOL; improves target encoder quality over epochs).
#   4. Gradient norm clipping (max_norm=1.0) for training stability.
#
# Datasets: MUTAG, PROTEINS, IMDB-BINARY, IMDB-MULTI, DD, ZINC.
# Estimated wall time on H100: ~80 min total.
# ============================================================

# ─────────────────────────────────────────────────────────────
# 0. ENVIRONMENT SETUP
# ─────────────────────────────────────────────────────────────
import os, sys, subprocess, time, math

REPO_URL = "https://github.com/trungkiet2005/graph_jepa_ml.git"
REPO_DIR = "/kaggle/working/graph_jepa_ml"

if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
else:
    subprocess.run(["git", "-C", REPO_DIR, "pull", "--ff-only"], check=True)

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

from core.config   import cfg, update_cfg
from core.get_data import create_dataset
from core.get_model import create_model
from torch_geometric.loader import DataLoader
from core.trainer import run_k_fold, run, k_fold
from core.tracker_footer import print_exp_tracker_footer
from core.model import GraphHMSJepa

# ─────────────────────────────────────────────────────────────
# 2. GLOBAL GPU SETTINGS
# ─────────────────────────────────────────────────────────────
torch.backends.cudnn.benchmark = True
USE_AMP   = torch.cuda.is_available()
AMP_DTYPE = torch.bfloat16

# ─────────────────────────────────────────────────────────────
# 3. EXP-09 HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────
EMA_START_M  = 0.996   # EMA momentum at epoch 0  (BYOL/I-JEPA default)
EMA_END_M    = 1.0     # EMA momentum at final epoch (asymptotically stop updating target)
VIC_START_W  = 0.05    # VICReg variance weight at epoch 0
VIC_END_W    = 0.0     # VICReg variance weight at final epoch (cosine anneal to 0)
GRAD_CLIP    = 1.0     # Max gradient norm (disabled when 0.0)

# Global state: trainer passes `sharp` arg but we repurpose it for the
# epoch counter so we can compute cosine schedules inside train().
_CURRENT_EPOCH = [0]
_TOTAL_EPOCHS  = [50]  # overwritten per dataset before run_k_fold

# ─────────────────────────────────────────────────────────────
# 4. PER-DATASET CONFIGS
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


# ─────────────────────────────────────────────────────────────
# 5. HELPERS
# ─────────────────────────────────────────────────────────────

def _device_tensor(cfg):
    d = cfg.device
    if isinstance(d, torch.device): return d
    if isinstance(d, str):          return torch.device(d)
    if isinstance(d, int):          return torch.device(f"cuda:{d}" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def _merge_cfg_for_dataset(dataset_name):
    cfg_yaml = DATASET_CONFIGS[dataset_name]
    cfg_path = f"/tmp/exp09_{dataset_name.replace('-', '_').lower()}.yaml"
    with open(cfg_path, "w") as f:
        f.write(cfg_yaml)
    from core.config import cfg as _cfg
    _cfg.defrost()
    _cfg.merge_from_file(cfg_path)
    updated_cfg = update_cfg(_cfg, args_str="")
    updated_cfg.k = 10
    return updated_cfg


def _cosine_schedule(start, end, current_epoch, total_epochs):
    """Cosine annealing from `start` to `end` over `total_epochs` epochs."""
    if total_epochs <= 1:
        return end
    t = current_epoch / (total_epochs - 1)            # 0 → 1
    return end + (start - end) * (1 + math.cos(math.pi * t)) / 2


# ─────────────────────────────────────────────────────────────
# 6. COSINE LOSS + ANNEALED VICReg
# ─────────────────────────────────────────────────────────────

def _cosine_jepa_loss(model, data, vic_weight: float):
    """
    Compute cosine-similarity JEPA loss across all three HMS scales.
    Returns (loss, num_targets).
    """
    if isinstance(model, GraphHMSJepa):
        (tx0, ty0), (tx1, ty1), (tx2, ty2) = model(data)
        # Negative cosine similarity (mean over patch tokens and batch)
        cos_l0 = 1.0 - F.cosine_similarity(ty0, tx0.detach(), dim=-1)
        cos_l1 = 1.0 - F.cosine_similarity(ty1, tx1.detach(), dim=-1)
        cos_l2 = 1.0 - F.cosine_similarity(ty2, tx2.detach(), dim=-1)
        w = model.loss_weights
        loss = w[0] * cos_l0.mean() + w[1] * cos_l1.mean() + w[2] * cos_l2.mean()
        # VICReg variance hinge (annealed weight)
        if vic_weight > 0:
            loss = loss + vic_weight * torch.mean(
                torch.relu(1.0 - tx0.detach().std(dim=0)))
        num_t = len(ty0)
    else:
        target_x, target_y = model(data)
        cos_loss = 1.0 - F.cosine_similarity(target_y, target_x.detach(), dim=-1)
        loss = cos_loss.mean()
        if vic_weight > 0:
            loss = loss + vic_weight * torch.mean(
                torch.relu(1.0 - target_x.detach().std(dim=0)))
        num_t = len(target_y)
    return loss, num_t


def _ema_update_cosine(model, momentum_weight):
    """EMA update with cosine-scheduled momentum."""
    if isinstance(model, GraphHMSJepa):
        for ctx_enc, tgt_enc in model.encoder_pairs:
            for pq, pk in zip(ctx_enc.parameters(), tgt_enc.parameters()):
                pk.data.mul_(momentum_weight).add_((1. - momentum_weight) * pq.detach().data)
    else:
        for pq, pk in zip(model.context_encoder.parameters(),
                          model.target_encoder.parameters()):
            pk.data.mul_(momentum_weight).add_((1. - momentum_weight) * pq.detach().data)


# ─────────────────────────────────────────────────────────────
# 7. PREFLIGHT
# ─────────────────────────────────────────────────────────────

def preflight_one_dataset(dataset_name):
    cfg = _merge_cfg_for_dataset(dataset_name)
    device = _device_tensor(cfg)

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
        loader = DataLoader(small, batch_size=bs, shuffle=True,
                            num_workers=0, pin_memory=False)
    else:
        dataset, transform, _ = out
        train_indices, _ = k_fold(dataset, cfg.k)
        ti  = train_indices[0]
        n   = min(_PREFLIGHT_MAX_GRAPHS, len(ti))
        if n < 1:
            raise RuntimeError("Preflight fold has no training indices")
        subset = dataset[ti[:n]]
        subset.transform = transform
        train_list = [x for x in subset] if not cfg.metis.online else subset
        bs = min(cfg.train.batch_size, max(1, len(train_list)))
        loader = DataLoader(train_list, batch_size=bs, shuffle=True,
                            num_workers=0, pin_memory=False)

    model     = create_model(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr,
                                 weight_decay=getattr(cfg.train, 'wd', 0))
    scaler    = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    data      = next(iter(loader))
    if getattr(model, "use_lap", False) and hasattr(data, "lap_pos_enc") and data.lap_pos_enc is not None:
        b_pe = data.lap_pos_enc
        sf   = torch.rand(b_pe.size(1)); sf[sf >= 0.5] = 1.0; sf[sf < 0.5] = -1.0
        data.lap_pos_enc = b_pe * sf.unsqueeze(0)
    data = data.to(device)
    optimizer.zero_grad()
    with torch.amp.autocast("cuda", enabled=USE_AMP, dtype=AMP_DTYPE):
        loss, _ = _cosine_jepa_loss(model, data, VIC_START_W)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()


def run_preflight_all():
    print("\n" + "=" * 70)
    print("  EXP 09 — PREFLIGHT (Cosine Loss + Annealed VICReg)")
    print("=" * 70)
    t0 = time.time()
    for name in DATASETS_TO_RUN:
        t_ds = time.time()
        print(f"\n  [preflight] {name} ...", flush=True)
        preflight_one_dataset(name)
        print(f"  [preflight] {name} OK ({time.time() - t_ds:.1f}s)", flush=True)
    print(f"\n  Preflight finished in {(time.time() - t0) / 60.0:.1f} min.\n")


# ─────────────────────────────────────────────────────────────
# 8. TRAIN / TEST LOOPS
# ─────────────────────────────────────────────────────────────

def train(train_loader, model, optimizer, evaluator, device,
          momentum_weight, sharp=None, criterion_type=0):
    """
    `sharp` is repurposed as a dict with keys 'epoch' and 'total_epochs'
    to compute cosine schedules. Falls back gracefully if not provided.
    """
    epoch       = (sharp or {}).get('epoch', 0)
    total_ep    = (sharp or {}).get('total_epochs', 50)
    vic_w       = _cosine_schedule(VIC_START_W, VIC_END_W, epoch, total_ep)
    ema_m       = _cosine_schedule(EMA_START_M, EMA_END_M, epoch, total_ep)

    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    step_losses, num_targets = [], []

    for data in train_loader:
        if model.use_lap:
            b_pe = data.lap_pos_enc
            sf   = torch.rand(b_pe.size(1)); sf[sf >= 0.5] = 1.0; sf[sf < 0.5] = -1.0
            data.lap_pos_enc = b_pe * sf.unsqueeze(0)
        data = data.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=USE_AMP, dtype=AMP_DTYPE):
            loss, num_t = _cosine_jepa_loss(model, data, vic_w)

        scaler.scale(loss).backward()
        if GRAD_CLIP > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        step_losses.append(loss.item())
        num_targets.append(num_t)

        with torch.no_grad():
            _ema_update_cosine(model, ema_m)

    if epoch % 5 == 0:
        print(f"  [Epoch {epoch}] vic_w={vic_w:.4f}  ema_m={ema_m:.5f}", flush=True)

    return None, np.average(step_losses, weights=num_targets)


@torch.no_grad()
def test(loader, model, evaluator, device, criterion_type=0):
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    step_losses, num_targets = [], []
    for data in loader:
        data = data.to(device)
        with torch.amp.autocast('cuda', enabled=USE_AMP, dtype=AMP_DTYPE):
            loss, num_t = _cosine_jepa_loss(model, data, 0.0)
        step_losses.append(loss.item())
        num_targets.append(num_t)
    return None, np.average(step_losses, weights=num_targets)


# ─────────────────────────────────────────────────────────────
# 9. ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    skip_pf = os.environ.get("EXP09_SKIP_PREFLIGHT", "0") == "1"
    pf_only = os.environ.get("EXP09_PREFLIGHT_ONLY", "0") == "1"

    print(f"\n  EXP 09: Cosine Loss + Annealed VICReg + Cosine EMA Schedule")
    print(f"  VICReg weight: {VIC_START_W} → {VIC_END_W}  |  EMA momentum: {EMA_START_M} → {EMA_END_M}")
    print(f"  Grad clip: {GRAD_CLIP}")

    if not skip_pf:
        try:
            run_preflight_all()
        except Exception as e:
            print(f"\n  PREFLIGHT FAILED: {e!r}", flush=True)
            raise
        if pf_only:
            print("EXP09_PREFLIGHT_ONLY=1 — exiting after successful preflight.")
            sys.exit(0)

    wall_times = {}
    tracker_results = {}
    total_start = time.time()

    for dataset_name in DATASETS_TO_RUN:
        ds_start = time.time()
        print(f"\n{'#' * 70}")
        print(f"#  EXP 09 — Cosine Loss — DATASET: {dataset_name}")
        print(f"{'#' * 70}")

        updated_cfg = _merge_cfg_for_dataset(dataset_name)
        _TOTAL_EPOCHS[0] = updated_cfg.train.epochs   # expose for scheduler

        # Inject epoch counter via closure over run_k_fold's `sharp` param
        # run_k_fold signature: run_k_fold(cfg, create_dataset, create_model, train, test)
        # It internally calls train(..., sharp=None) — we patch to pass epoch info
        import functools
        _epoch_counter = [0]

        def _train_with_schedule(train_loader, model, optimizer, evaluator, device,
                                 momentum_weight, sharp=None, criterion_type=0):
            sharp = {'epoch': _epoch_counter[0], 'total_epochs': updated_cfg.train.epochs}
            _epoch_counter[0] += 1
            return train(train_loader, model, optimizer, evaluator, device,
                         momentum_weight, sharp, criterion_type)

        is_regression = (dataset_name == "ZINC")
        if is_regression:
            tracker_results[dataset_name] = run(
                updated_cfg, create_dataset, create_model, _train_with_schedule, test
            )
        else:
            tracker_results[dataset_name] = run_k_fold(
                updated_cfg, create_dataset, create_model, _train_with_schedule, test
            )

        ds_elapsed = time.time() - ds_start
        wall_times[dataset_name] = ds_elapsed
        print(f"\n  [{dataset_name}] wall time: {ds_elapsed / 60:.1f} min")

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"  EXP 09 (Cosine Loss + Sched.) — ALL DATASETS COMPLETE")
    print(f"  Total wall time: {total_elapsed / 60:.1f} min")
    print(f"{'=' * 70}")
    for ds, t in wall_times.items():
        print(f"  {ds:<12}  {t / 60:>9.1f}m")
    print(f"{'=' * 70}")
    print_exp_tracker_footer(
        9,
        "Cosine Loss + Annealed VICReg + Cosine EMA Schedule",
        tracker_results,
    )
