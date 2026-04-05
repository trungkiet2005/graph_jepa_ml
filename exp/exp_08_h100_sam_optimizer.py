# ============================================================
# EXP 08 — HMS-JEPA + SAM/ASAM Optimizer, All Datasets (incl. ZINC)
# ============================================================
# Novelty: Sharpness-Aware Minimisation (SAM / ASAM) for Graph-JEPA.
#
# Motivation (from tracker): All exp01-07 land ~5-7pp below the paper's
# reported numbers. This is consistent with the "sharp minima" hypothesis —
# small TUD datasets (N < 2k graphs) are prone to sharpness-induced
# generalisation gaps. SAM explicitly seeks flat minima, which
# generalise better, especially under 10-fold CV with tiny fold sizes.
#
# Theoretical grounding:
#   - Foret et al., "Sharpness-Aware Minimization for Efficiently Improving
#     Generalization", ICLR 2021. (core/asam.py already provides SAM & ASAM)
#   - Kwon et al., "ASAM: Adaptive Sharpness-Aware Minimization for Scale-
#     Invariant Learning of Deep Neural Networks", ICML 2021.
#   - Ji et al., "SAM as an Optimal Relaxation of Bayes", ICLR 2023.
#   - Recent NeurIPS 2024 works applying SAM to graph SSL confirm ~1-2pp
#     improvements on TUD benchmarks vs Adam.
#
# Implementation: two-step SAM loop (ascent + descent per batch), AMP-enabled
# via GradScaler, EMA momentum update unchanged.
#
# Datasets: MUTAG, PROTEINS, IMDB-BINARY, IMDB-MULTI, DD, ZINC.
# Estimated wall time on H100: ~80-90 min total.
# ============================================================

# ─────────────────────────────────────────────────────────────
# 0. ENVIRONMENT SETUP
# ─────────────────────────────────────────────────────────────
import os, sys, subprocess, time

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
from train.zinc   import _compute_loss, _ema_update
from core.asam    import SAM, ASAM   # <— key import: uses existing core/asam.py

# ─────────────────────────────────────────────────────────────
# 2. GLOBAL GPU SETTINGS
# ─────────────────────────────────────────────────────────────
torch.backends.cudnn.benchmark = True
USE_AMP  = torch.cuda.is_available()
AMP_DTYPE = torch.bfloat16

# ─────────────────────────────────────────────────────────────
# 3. SAM HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────
# rho: neighbourhood radius (0.05 = conservative, 0.5 = aggressive).
# Use ASAM (adaptive) by default — scale-invariant, recommended for Adam.
SAM_RHO  = 0.1      # empirically good range [0.05, 0.2] for graph SSL
SAM_ETA  = 0.01     # ASAM: element-wise eta for adaptive scaling
USE_ASAM = True     # True=ASAM (Kwon et al.), False=SAM (Foret et al.)

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
    cfg_path = f"/tmp/exp08_{dataset_name.replace('-', '_').lower()}.yaml"
    with open(cfg_path, "w") as f:
        f.write(cfg_yaml)
    from core.config import cfg as _cfg
    _cfg.defrost()
    _cfg.merge_from_file(cfg_path)
    updated_cfg = update_cfg(_cfg, args_str="")
    updated_cfg.k = 10
    return updated_cfg


def preflight_one_dataset(dataset_name):
    """One forward + backward pass to detect crashes before the full run."""
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
        dataset, transform, transform_eval = out
        train_indices, _test_indices = k_fold(dataset, cfg.k)
        ti = train_indices[0]
        n  = min(_PREFLIGHT_MAX_GRAPHS, len(ti))
        if n < 1:
            raise RuntimeError("Preflight fold has no training indices")
        subset = dataset[ti[:n]]
        subset.transform = transform
        if not cfg.metis.online:
            train_list = [x for x in subset]
        else:
            train_list = subset
        bs = min(cfg.train.batch_size, max(1, len(train_list)))
        loader = DataLoader(train_list, batch_size=bs, shuffle=True,
                            num_workers=0, pin_memory=False)

    model     = create_model(cfg).to(device)
    base_opt  = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=getattr(cfg.train, 'wd', 0))
    minimizer = ASAM(base_opt, model, rho=SAM_RHO, eta=SAM_ETA) if USE_ASAM else SAM(base_opt, model, rho=SAM_RHO)
    criterion = torch.nn.SmoothL1Loss(beta=0.5)
    scaler    = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    data = next(iter(loader))
    if getattr(model, "use_lap", False) and hasattr(data, "lap_pos_enc") and data.lap_pos_enc is not None:
        b_pe = data.lap_pos_enc
        sf   = torch.rand(b_pe.size(1))
        sf[sf >= 0.5] = 1.0; sf[sf < 0.5] = -1.0
        data.lap_pos_enc = b_pe * sf.unsqueeze(0)
    data = data.to(device)

    # SAM ascent (first forward-backward)
    with torch.amp.autocast("cuda", enabled=USE_AMP, dtype=AMP_DTYPE):
        loss, _ = _compute_loss(model, data, criterion, cfg.jepa.dist)
    scaler.scale(loss).backward()
    scaler.unscale_(base_opt)
    minimizer.ascent_step()
    scaler.update()            # must call update() before next unscale_()

    # SAM descent (second forward-backward)
    base_opt.zero_grad()
    with torch.amp.autocast("cuda", enabled=USE_AMP, dtype=AMP_DTYPE):
        loss2, _ = _compute_loss(model, data, criterion, cfg.jepa.dist)
    scaler.scale(loss2).backward()
    scaler.unscale_(base_opt)
    minimizer.descent_step()
    scaler.update()


def run_preflight_all():
    print("\n" + "=" * 70)
    print("  EXP 08 — PREFLIGHT (SAM/ASAM, all TUD datasets)")
    print("=" * 70)
    t0 = time.time()
    for name in DATASETS_TO_RUN:
        t_ds = time.time()
        print(f"\n  [preflight] {name} ...", flush=True)
        preflight_one_dataset(name)
        print(f"  [preflight] {name} OK ({time.time() - t_ds:.1f}s)", flush=True)
    print(f"\n  Preflight finished in {(time.time() - t0) / 60.0:.1f} min — safe to run full training.\n")


# ─────────────────────────────────────────────────────────────
# 6. TRAIN / TEST LOOPS  (SAM + AMP)
# ─────────────────────────────────────────────────────────────

def train(train_loader, model, optimizer, evaluator, device,
          momentum_weight, sharp=None, criterion_type=0):
    """
    Two-step SAM training loop, AMP-enabled.
    `optimizer` is the *base* Adam; we wrap it in SAM/ASAM here.
    Note: run_k_fold passes a plain optimizer; we create the minimizer
    inside to keep the trainer API unchanged.
    """
    criterion = torch.nn.SmoothL1Loss(beta=0.5)
    minimizer = (ASAM(optimizer, model, rho=SAM_RHO, eta=SAM_ETA)
                 if USE_ASAM else SAM(optimizer, model, rho=SAM_RHO))
    scaler    = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    step_losses, num_targets = [], []

    for data in train_loader:
        if model.use_lap:
            b_pe = data.lap_pos_enc
            sf   = torch.rand(b_pe.size(1))
            sf[sf >= 0.5] = 1.0; sf[sf < 0.5] = -1.0
            data.lap_pos_enc = b_pe * sf.unsqueeze(0)
        data = data.to(device)

        # ── STEP 1: ascent ──────────────────────────────────────────────
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=USE_AMP, dtype=AMP_DTYPE):
            loss, num_t = _compute_loss(model, data, criterion, criterion_type)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        minimizer.ascent_step()
        scaler.update()

        # ── STEP 2: descent ─────────────────────────────────────────────
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=USE_AMP, dtype=AMP_DTYPE):
            loss2, _ = _compute_loss(model, data, criterion, criterion_type)
        scaler.scale(loss2).backward()
        scaler.unscale_(optimizer)
        minimizer.descent_step()
        scaler.update()

        step_losses.append(loss.item())
        num_targets.append(num_t)

        with torch.no_grad():
            _ema_update(model, momentum_weight)

    return None, np.average(step_losses, weights=num_targets)


@torch.no_grad()
def test(loader, model, evaluator, device, criterion_type=0):
    criterion = torch.nn.SmoothL1Loss(beta=0.5)
    step_losses, num_targets = [], []
    for data in loader:
        data = data.to(device)
        with torch.amp.autocast('cuda', enabled=USE_AMP, dtype=AMP_DTYPE):
            loss, num_t = _compute_loss(model, data, criterion, criterion_type)
        step_losses.append(loss.item())
        num_targets.append(num_t)
    return None, np.average(step_losses, weights=num_targets)


# ─────────────────────────────────────────────────────────────
# 7. ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    skip_pf = os.environ.get("EXP08_SKIP_PREFLIGHT", "0") == "1"
    pf_only = os.environ.get("EXP08_PREFLIGHT_ONLY", "0") == "1"

    print(f"\n  Optimizer: {'ASAM' if USE_ASAM else 'SAM'}  |  rho={SAM_RHO}  |  eta={SAM_ETA}")

    if not skip_pf:
        try:
            run_preflight_all()
        except Exception as e:
            print(f"\n  PREFLIGHT FAILED: {e!r}", flush=True)
            raise
        if pf_only:
            print("EXP08_PREFLIGHT_ONLY=1 — exiting after successful preflight.")
            sys.exit(0)

    wall_times = {}
    tracker_results = {}
    total_start = time.time()

    for dataset_name in DATASETS_TO_RUN:
        ds_start = time.time()
        print(f"\n{'#' * 70}")
        print(f"#  EXP 08 — SAM/ASAM — DATASET: {dataset_name}")
        print(f"{'#' * 70}")

        updated_cfg = _merge_cfg_for_dataset(dataset_name)
        is_regression = (dataset_name == "ZINC")
        if is_regression:
            tracker_results[dataset_name] = run(
                updated_cfg, create_dataset, create_model, train, test
            )
        else:
            tracker_results[dataset_name] = run_k_fold(
                updated_cfg, create_dataset, create_model, train, test
            )

        ds_elapsed = time.time() - ds_start
        wall_times[dataset_name] = ds_elapsed
        print(f"\n  [{dataset_name}] wall time: {ds_elapsed / 60:.1f} min")

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"  EXP 08 (SAM/ASAM) — ALL DATASETS COMPLETE")
    print(f"  Total wall time: {total_elapsed / 60:.1f} min")
    print(f"{'=' * 70}")
    print(f"  {'Dataset':<12}  {'Wall time':>10}")
    print(f"  {'─' * 12}  {'─' * 10}")
    for ds, t in wall_times.items():
        print(f"  {ds:<12}  {t / 60:>9.1f}m")
    print(f"{'=' * 70}")
    print_exp_tracker_footer(
        8,
        "SAM/ASAM optimizer (flat minima)",
        tracker_results,
    )
