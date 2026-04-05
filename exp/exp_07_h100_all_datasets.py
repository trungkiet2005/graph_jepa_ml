# ============================================================
# EXP 07 — HMS-JEPA Baseline, All Fast Datasets  (Kaggle H100-ready)
# ============================================================
# Runs MUTAG, PROTEINS, IMDB-BINARY, IMDB-MULTI, DD, ZINC sequentially.
# Estimated total wall-clock time on H100 80GB: < 2 hours.
#
# Key H100 optimisations applied here:
#   1. metis.online = False  — pre-compute METIS once per fold  (93% CPU savings)
#   2. AMP bfloat16 mixed precision
#   3. torch.backends.cudnn.benchmark = True
#   4. pin_memory + persistent_workers in DataLoader  (trainer.py already updated)
#   5. Per-dataset tuned batch sizes (larger batch = more GPU utilisation)
#
# Estimated runtimes (online=False, H100):
#   MUTAG     ~  5 min   (188 graphs, 5 runs × 10 folds × 50 epochs)
#   PROTEINS  ~ 12 min   (1113 graphs, 5r × 10f × 30 epochs)
#   IMDB-BINARY ~ 10 min   (1000 graphs, 5r × 10f × 10 epochs)
#   IMDB-MULTI  ~  7 min   (1500 graphs, 5r × 10f ×  5 epochs)
#   DD        ~ 35 min   (1178 large graphs, 5r × 10f × 20 epochs)
#   ZINC      ~ 40 min   (12000 graphs, 10 runs × 30 epochs)
#   ──────────────────────────────────────────────────────
#   Total     ~109 min   ≈ 1h 49min
# ============================================================
#
# Preflight (recommended on Kaggle before a long run):
#   Default: run a quick check for every dataset in DATASETS_TO_RUN (load data,
#   METIS/offline transforms on a tiny subset, one AMP train step), then run the
#   full experiment. If anything fails, the script exits before wasting GPU time.
#   EXP07_SKIP_PREFLIGHT=1   — skip the check, run full only (reuse after a green preflight).
#   EXP07_PREFLIGHT_ONLY=1   — only run the check, then exit 0 (dry run).
#   EXP07_PREFLIGHT_ZINC_FAST=1 — ZINC preflight only: use metis.online=True so the check
#       finishes quickly; full run still uses offline METIS from the YAML (default ZINC
#       preflight follows the YAML and can take a long time on first METIS pass).
# ============================================================

# ─────────────────────────────────────────────────────────────
# 0. ENVIRONMENT SETUP
# ─────────────────────────────────────────────────────────────
import os, sys, subprocess, time, textwrap

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
from train.zinc    import _compute_loss, _ema_update

# ─────────────────────────────────────────────────────────────
# 2. GLOBAL GPU SETTINGS
# ─────────────────────────────────────────────────────────────
torch.backends.cudnn.benchmark = True
USE_AMP = torch.cuda.is_available()
AMP_DTYPE = torch.bfloat16  # H100 has native bfloat16 support

# ─────────────────────────────────────────────────────────────
# 3. PER-DATASET CONFIGS
#    All values from train/configs/<dataset>_hms.yaml
#    + online=False and larger batch sizes for H100
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

# Datasets to run — comment out any you want to skip
DATASETS_TO_RUN = ["MUTAG", "PROTEINS", "IMDB-BINARY", "IMDB-MULTI", "DD", "ZINC"]

# Max graphs to materialize per dataset during preflight (keeps METIS offline cost bounded)
_PREFLIGHT_MAX_GRAPHS = 24


def _device_tensor(cfg):
    d = cfg.device
    if isinstance(d, torch.device):
        return d
    if isinstance(d, str):
        return torch.device(d)
    if isinstance(d, int):
        return torch.device(f"cuda:{d}" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def _merge_cfg_for_dataset(dataset_name):
    cfg_yaml = DATASET_CONFIGS[dataset_name]
    cfg_path = f"/tmp/exp07_{dataset_name.replace('-', '_').lower()}.yaml"
    with open(cfg_path, "w") as f:
        f.write(cfg_yaml)
    from core.config import cfg as _cfg
    _cfg.defrost()
    _cfg.merge_from_file(cfg_path)
    updated_cfg = update_cfg(_cfg, args_str="")
    updated_cfg.k = 10
    return updated_cfg


def preflight_one_dataset(dataset_name):
    """Load data + one train step (forward/backward) for this dataset. Raises on failure."""
    cfg = _merge_cfg_for_dataset(dataset_name)
    if (
        cfg.dataset == "ZINC"
        and os.environ.get("EXP07_PREFLIGHT_ZINC_FAST", "0") == "1"
    ):
        cfg.defrost()
        cfg.metis.online = True
        cfg.freeze()
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
        loader = DataLoader(
            small, batch_size=bs, shuffle=True,
            num_workers=0, pin_memory=False,
        )
    else:
        dataset, transform, transform_eval = out
        train_indices, _test_indices = k_fold(dataset, cfg.k)
        ti = train_indices[0]
        n = min(_PREFLIGHT_MAX_GRAPHS, len(ti))
        if n < 1:
            raise RuntimeError("Preflight fold has no training indices")
        subset = dataset[ti[:n]]
        subset.transform = transform
        if not cfg.metis.online:
            train_list = [x for x in subset]
        else:
            train_list = subset
        bs = min(cfg.train.batch_size, max(1, len(train_list)))
        loader = DataLoader(
            train_list, batch_size=bs, shuffle=True,
            num_workers=0, pin_memory=False,
        )

    model = create_model(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.wd)
    criterion = torch.nn.SmoothL1Loss(beta=0.5)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    data = next(iter(loader))
    if getattr(model, "use_lap", False) and hasattr(data, "lap_pos_enc") and data.lap_pos_enc is not None:
        batch_pos_enc = data.lap_pos_enc
        sign_flip = torch.rand(batch_pos_enc.size(1))
        sign_flip[sign_flip >= 0.5] = 1.0
        sign_flip[sign_flip < 0.5] = -1.0
        data.lap_pos_enc = batch_pos_enc * sign_flip.unsqueeze(0)
    data = data.to(device)
    optimizer.zero_grad()
    with torch.amp.autocast("cuda", enabled=USE_AMP, dtype=AMP_DTYPE):
        loss, _num_t = _compute_loss(model, data, criterion, cfg.jepa.dist)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()


def run_preflight_all():
    print("\n" + "=" * 70)
    print("  EXP 07 — PREFLIGHT (all datasets in DATASETS_TO_RUN)")
    print("=" * 70)
    t0 = time.time()
    for name in DATASETS_TO_RUN:
        t_ds = time.time()
        print(f"\n  [preflight] {name} ...", flush=True)
        preflight_one_dataset(name)
        print(f"  [preflight] {name} OK ({time.time() - t_ds:.1f}s)", flush=True)
    print(f"\n  Preflight finished in {(time.time() - t0) / 60.0:.1f} min — safe to run full training.\n")


# ─────────────────────────────────────────────────────────────
# 4. TRAIN / TEST LOOPS  (AMP-enabled)
# ─────────────────────────────────────────────────────────────

def train(train_loader, model, optimizer, evaluator, device,
          momentum_weight, sharp=None, criterion_type=0):
    criterion = torch.nn.SmoothL1Loss(beta=0.5)
    step_losses, num_targets = [], []
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

    for data in train_loader:
        if model.use_lap:
            batch_pos_enc = data.lap_pos_enc
            sign_flip = torch.rand(batch_pos_enc.size(1))
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            data.lap_pos_enc = batch_pos_enc * sign_flip.unsqueeze(0)
        data = data.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=USE_AMP, dtype=AMP_DTYPE):
            loss, num_t = _compute_loss(model, data, criterion, criterion_type)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
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
# 5. ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    skip_pf = os.environ.get("EXP07_SKIP_PREFLIGHT", "0") == "1"
    pf_only = os.environ.get("EXP07_PREFLIGHT_ONLY", "0") == "1"

    if not skip_pf:
        try:
            run_preflight_all()
        except Exception as e:
            print(f"\n  PREFLIGHT FAILED: {e!r}", flush=True)
            raise
        if pf_only:
            print("EXP07_PREFLIGHT_ONLY=1 — exiting after successful preflight.")
            sys.exit(0)

    tracker_results = {}
    wall_times = {}

    total_start = time.time()

    for dataset_name in DATASETS_TO_RUN:
        ds_start = time.time()
        print(f"\n{'#'*70}")
        print(f"#  DATASET: {dataset_name}")
        print(f"{'#'*70}")

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
        print(f"\n  [{dataset_name}] wall time: {ds_elapsed/60:.1f} min")

    total_elapsed = time.time() - total_start

    print(f"\n{'='*70}")
    print(f"  EXP 07 — ALL DATASETS COMPLETE")
    print(f"  Total wall time: {total_elapsed/60:.1f} min")
    print(f"{'='*70}")
    print(f"  {'Dataset':<12}  {'Wall time':>10}")
    print(f"  {'─'*12}  {'─'*10}")
    for ds, t in wall_times.items():
        print(f"  {ds:<12}  {t/60:>9.1f}m")
    print(f"{'='*70}")
    print_exp_tracker_footer(7, "H100 all-datasets baseline (HMS-JEPA)", tracker_results)
