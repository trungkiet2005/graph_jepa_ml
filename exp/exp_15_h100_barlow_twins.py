# ============================================================
# EXP 15 — MCMT-JEPA + Barlow Twins Redundancy Reduction + Cosine LR Warmup
# ============================================================
# Novelty: Replace NT-Xent (EXP 13) with Barlow Twins cross-correlation
# redundancy reduction — batch-size invariant and aligned with JEPA.
# Add cosine LR warmup (5 epochs: lr/20 → lr, then cosine decay).
#
# Theoretical grounding:
#   - Zbontar et al., "Barlow Twins", NeurIPS 2021.
#   - Ma et al., "Graph Barlow Twins (BT-GCL)", AAAI 2023.
#   - Assran et al., "I-JEPA", CVPR 2023 (multi-context base).
#
# Key changes vs EXP 13:
#   1. barlow_twins_loss() replaces nt_xent_loss()
#      C[i,j] = mean(z1_i * z2_j), L = Σ(1-C_ii)² + λ_BT*Σ_{i≠j} C_ij²
#   2. Proj head: 512→512→512 (wider than EXP 13's 512→128)
#   3. Cosine LR warmup: 5 epochs warmup then cosine decay
#   4. Extended epochs per dataset for better convergence
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
from train.zinc import _ema_update

torch.backends.cudnn.benchmark = True
USE_AMP   = torch.cuda.is_available()
AMP_DTYPE = torch.bfloat16

NUM_CONTEXTS     = 2
BT_WEIGHT        = 0.05
BT_LAMBDA        = 0.005
BT_PROJ_DIM      = 512
LR_WARMUP_EPOCHS = 5


# ─────────────────────────────────────────────────────────────
# BARLOW TWINS LOSS
# ─────────────────────────────────────────────────────────────
def _off_diagonal(x):
    n, m = x.shape; assert n == m
    return x.flatten()[:-1].view(n-1, n+1)[:, 1:].flatten()

def barlow_twins_loss(z1, z2, lambda_bt=BT_LAMBDA):
    B, D = z1.shape
    z1_n = (z1 - z1.mean(0)) / (z1.std(0) + 1e-5)
    z2_n = (z2 - z2.mean(0)) / (z2.std(0) + 1e-5)
    C = torch.mm(z1_n.T, z2_n) / B
    on_diag  = torch.diagonal(C).add_(-1).pow_(2).sum()
    off_diag = _off_diagonal(C).pow_(2).sum()
    return on_diag + lambda_bt * off_diag


# ─────────────────────────────────────────────────────────────
# MULTI-CONTEXT CROSS-ATTENTION MODULE
# ─────────────────────────────────────────────────────────────
class MultiContextAggregator(nn.Module):
    def __init__(self, nhid: int, nhead: int = 8):
        super().__init__()
        self.ctx_self_attn = nn.MultiheadAttention(nhid, nhead, batch_first=True)
        self.ctx_norm1     = nn.LayerNorm(nhid)
        self.cross_attn    = nn.MultiheadAttention(nhid, nhead, batch_first=True)
        self.cross_norm    = nn.LayerNorm(nhid)
        self.ffn  = nn.Sequential(nn.Linear(nhid, nhid*2), nn.GELU(), nn.Linear(nhid*2, nhid))
        self.norm2 = nn.LayerNorm(nhid)

    def forward(self, ctx_x, tgt_pe):
        ctx_sa, _ = self.ctx_self_attn(ctx_x, ctx_x, ctx_x)
        ctx_x = self.ctx_norm1(ctx_x + ctx_sa)
        ctx_ca, _ = self.cross_attn(tgt_pe, ctx_x, ctx_x)
        out = self.cross_norm(tgt_pe + ctx_ca)
        return self.norm2(out + self.ffn(out))


# ─────────────────────────────────────────────────────────────
# MODEL: MCMT-JEPA + BARLOW TWINS
# ─────────────────────────────────────────────────────────────
class GraphHMSJepaBarlowTwins(GraphHMSJepa):
    def __init__(self, *args, num_contexts=2, proj_dim=BT_PROJ_DIM, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_contexts = num_contexts
        nhid = self.nhid
        self.mc_agg_L0 = MultiContextAggregator(nhid)
        self.mc_agg_L1 = MultiContextAggregator(nhid)
        self.proj_head = nn.Sequential(
            nn.Linear(nhid, proj_dim), nn.BatchNorm1d(proj_dim), nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim), nn.BatchNorm1d(proj_dim), nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward_bt(self, data):
        device = data.y.device
        subgraph_x_L0, raw_patch_pes, _ = self._gnn_forward(data)
        subgraph_x_L1    = scatter(subgraph_x_L0, data.fine_to_medium, dim=0, reduce='mean')
        raw_patch_pes_L1 = scatter(raw_patch_pes,  data.fine_to_medium, dim=0, reduce='mean')
        subgraph_x_L2    = scatter(subgraph_x_L1, data.medium_to_coarse, dim=0, reduce='mean')
        raw_patch_pes_L2 = scatter(raw_patch_pes_L1, data.medium_to_coarse, dim=0, reduce='mean')
        B = len(data.call_n_patches)

        def _bi(npl):
            bi = torch.tensor(np.cumsum(npl))
            return torch.hstack((torch.tensor(0), bi[:-1])).to(device)

        bi_L0 = _bi(data.call_n_patches)
        bi_L1 = _bi(data.n_patches_L1)
        bi_L2 = _bi(data.n_patches_L2)

        # L0 contexts
        ctx_idx_L0  = data.context_subgraph_idx + bi_L0
        tgt_idx_L0  = torch.vstack([torch.tensor(dt) for dt in data.target_subgraph_idxs]).to(device)
        tgt_idx_L0 += bi_L0.unsqueeze(1)
        ctx_patch_L0 = subgraph_x_L0[ctx_idx_L0] + self.patch_rw_encoder(raw_patch_pes[ctx_idx_L0])
        n_ppg = torch.tensor([c[0] for c in data.call_n_patches], device=device)
        ctx2_idx_L0 = ((data.context_subgraph_idx + 1) % n_ppg) + bi_L0
        ctx2_patch_L0 = subgraph_x_L0[ctx2_idx_L0] + self.patch_rw_encoder(raw_patch_pes[ctx2_idx_L0])
        ctx_stack = torch.stack([ctx_patch_L0, ctx2_patch_L0], dim=1)  # [B,2,nhid]

        # Target L0
        tgt_pe_enc_L0 = self.patch_rw_encoder(raw_patch_pes[tgt_idx_L0.flatten()])
        tgt_x_L0 = subgraph_x_L0[tgt_idx_L0.flatten()].reshape(B, self.num_target_patches, self.nhid)
        with torch.no_grad():
            if hasattr(data, 'coarsen_adj'):
                tgt_rel = torch.vstack([torch.tensor(dt) for dt in data.target_subgraph_idxs])
                adj = data.coarsen_adj[torch.arange(B).unsqueeze(1).unsqueeze(2), tgt_rel.unsqueeze(1), tgt_rel.unsqueeze(2)]
                tgt_x_L0 = self.target_encoder_L0(tgt_x_L0, adj, None)
            else:
                tgt_x_L0 = self.target_encoder_L0(tgt_x_L0, None, None)
        pred_L0 = self.predictor_L0_to_L0(self.mc_agg_L0(ctx_stack, tgt_pe_enc_L0.reshape(B, self.num_target_patches, self.nhid)))

        # Barlow Twins views
        z1 = self.proj_head(self.context_encoder_L0(ctx_patch_L0.unsqueeze(1), None, None).squeeze(1))
        z2 = self.proj_head(self.context_encoder_L0(ctx2_patch_L0.unsqueeze(1), None, None).squeeze(1))

        # L0→L1
        tgt_idx_L1 = torch.vstack([torch.tensor(dt) for dt in data.target_subgraph_idxs_L1]).to(device) + bi_L1.unsqueeze(1)
        nT_L1 = tgt_idx_L1.shape[1]
        tgt_x_L1 = subgraph_x_L1[tgt_idx_L1.flatten()].reshape(B, nT_L1, self.nhid)
        with torch.no_grad():
            tgt_rel_L1 = torch.vstack([torch.tensor(dt) for dt in data.target_subgraph_idxs_L1])
            adj_L1 = data.coarsen_adj_L1[torch.arange(B).unsqueeze(1).unsqueeze(2), tgt_rel_L1.unsqueeze(1), tgt_rel_L1.unsqueeze(2)]
            tgt_x_L1 = self.target_encoder_L1(tgt_x_L1, adj_L1, None)
        tgt_pe_L1_r = self.patch_rw_encoder_L1(raw_patch_pes_L1[tgt_idx_L1.flatten()]).reshape(B, nT_L1, self.nhid)
        pred_L1 = self.predictor_L0_to_L1(self.mc_agg_L1(ctx_stack, tgt_pe_L1_r))

        # L1→L2
        ctx_idx_L1 = data.context_subgraph_idx_L1 + bi_L1
        ctx_patch_L1 = subgraph_x_L1[ctx_idx_L1] + self.patch_rw_encoder_L1(raw_patch_pes_L1[ctx_idx_L1])
        ctx_x_L1 = self.context_encoder_L1(ctx_patch_L1.unsqueeze(1), None, ~data.mask_L1.flatten()[ctx_idx_L1].reshape(B, 1))
        tgt_idx_L2 = torch.vstack([torch.tensor(dt) for dt in data.target_subgraph_idxs_L2]).to(device) + bi_L2.unsqueeze(1)
        nT_L2 = tgt_idx_L2.shape[1]
        tgt_x_L2 = subgraph_x_L2[tgt_idx_L2.flatten()].reshape(B, nT_L2, self.nhid)
        with torch.no_grad():
            tgt_rel_L2 = torch.vstack([torch.tensor(dt) for dt in data.target_subgraph_idxs_L2])
            adj_L2 = data.coarsen_adj_L2[torch.arange(B).unsqueeze(1).unsqueeze(2), tgt_rel_L2.unsqueeze(1), tgt_rel_L2.unsqueeze(2)]
            tgt_x_L2 = self.target_encoder_L2(tgt_x_L2, adj_L2, None)
        tgt_pe_L2_enc = self.patch_rw_encoder_L2(raw_patch_pes_L2[tgt_idx_L2.flatten()]).reshape(B, nT_L2, self.nhid)
        pred_L2 = self.predictor_L1_to_L2(ctx_x_L1 + tgt_pe_L2_enc)

        return (tgt_x_L0, pred_L0), (tgt_x_L1, pred_L1), (tgt_x_L2, pred_L2), (z1, z2)


# ─────────────────────────────────────────────────────────────
# LOSS
# ─────────────────────────────────────────────────────────────
def _bt_loss(model, data):
    if not isinstance(model, GraphHMSJepaBarlowTwins):
        from train.zinc import _compute_loss
        return _compute_loss(model, data, nn.SmoothL1Loss(beta=0.5), 0)
    criterion = nn.SmoothL1Loss(beta=0.5)
    (tx0, ty0), (tx1, ty1), (tx2, ty2), (z1, z2) = model.forward_bt(data)
    w = model.loss_weights
    loss = w[0]*criterion(ty0,tx0) + w[1]*criterion(ty1,tx1) + w[2]*criterion(ty2,tx2)
    loss = loss + model.var_weight * torch.mean(torch.relu(1.0 - tx0.detach().std(dim=0)))
    if z1.shape[0] > 1:
        loss = loss + BT_WEIGHT * barlow_twins_loss(z1, z2)
    return loss, len(ty0)


# ─────────────────────────────────────────────────────────────
# MODEL FACTORY
# ─────────────────────────────────────────────────────────────
def _resolve_dataset_types(name):
    m = {'MUTAG':('Linear','Linear',7,4),'PROTEINS':('Linear','Linear',3,1),
         'DD':('Linear','Linear',89,1),'IMDB-BINARY':('Linear','Linear',1,1),
         'IMDB-MULTI':('Linear','Linear',1,1),'REDDIT-BINARY':('Linear','Linear',1,1),
         'ZINC':('Discrete','Discrete',28,4)}
    return m.get(name, ('Linear','Linear',1,1))

def create_model_bt(cfg):
    base = create_model(cfg)
    if not isinstance(base, GraphHMSJepa): return base
    nt, et, nfn, nfe = _resolve_dataset_types(cfg.dataset)
    model = GraphHMSJepaBarlowTwins(
        nfeat_node=nfn, nfeat_edge=nfe, nhid=base.nhid, nout=1,
        nlayer_gnn=len(base.gnns), nlayer_mlpmixer=cfg.model.nlayer_mlpmixer,
        node_type=nt, edge_type=et, gnn_type=cfg.model.gnn_type, gMHA_type=cfg.model.gMHA_type,
        rw_dim=cfg.pos_enc.rw_dim, lap_dim=cfg.pos_enc.lap_dim,
        dropout=getattr(cfg.train,'dropout',0), mlpmixer_dropout=getattr(cfg.train,'mlpmixer_dropout',0),
        n_patches=cfg.metis.n_patches, patch_rw_dim=cfg.pos_enc.patch_rw_dim,
        num_context_patches=cfg.jepa.num_context, num_target_patches=cfg.jepa.num_targets,
        num_target_patches_L1=cfg.jepa.num_targets_L1, num_target_patches_L2=cfg.jepa.num_targets_L2,
        loss_weights=cfg.jepa.loss_weights, var_weight=cfg.jepa.var_weight,
        num_contexts=NUM_CONTEXTS, proj_dim=BT_PROJ_DIM,
    )
    p = sum(x.numel() for x in model.parameters() if x.requires_grad)
    print(f"  [BT Model] params={p/1e6:.2f}M  num_ctx={NUM_CONTEXTS}  proj_dim={BT_PROJ_DIM}  λ={BT_LAMBDA}")
    return model


# ─────────────────────────────────────────────────────────────
# DATASET CONFIGS
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
  epochs: 100
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
  epochs: 60
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
  epochs: 20
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
  epochs: 40
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
  epochs: 60
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

def _merge_cfg_for_dataset(ds):
    cfg_path = f"/tmp/exp15_{ds.replace('-','_').lower()}.yaml"
    with open(cfg_path, "w") as f: f.write(DATASET_CONFIGS[ds])
    from core.config import cfg as _cfg
    _cfg.defrost(); _cfg.merge_from_file(cfg_path)
    updated = update_cfg(_cfg, args_str=""); updated.k = 10
    return updated

def preflight_one_dataset(ds):
    cfg = _merge_cfg_for_dataset(ds); device = _device_tensor(cfg)
    out = create_dataset(cfg)
    if cfg.dataset == "ZINC":
        td, _, _ = out; n = min(_PREFLIGHT_MAX_GRAPHS, len(td))
        small = [x for x in td[:n]] if not cfg.metis.online else td[:n]
        loader = DataLoader(small, batch_size=min(cfg.train.batch_size, max(1,n)), shuffle=True, num_workers=0)
    else:
        dataset, transform, _ = out
        ti = k_fold(dataset, cfg.k)[0][0]; n = min(_PREFLIGHT_MAX_GRAPHS, len(ti))
        sub = dataset[ti[:n]]; sub.transform = transform
        loader = DataLoader([x for x in sub], batch_size=min(cfg.train.batch_size, max(1,n)), shuffle=True, num_workers=0)
    model = create_model_bt(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    data = next(iter(loader)).to(device); optimizer.zero_grad()
    with torch.amp.autocast("cuda", enabled=USE_AMP, dtype=AMP_DTYPE):
        loss, _ = _bt_loss(model, data)
    scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()

def run_preflight_all():
    print("\n" + "="*70)
    print("  EXP 15 — PREFLIGHT (MCMT-JEPA + Barlow Twins + Cosine LR Warmup)")
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
            loss, num_t = _bt_loss(model, data)
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
            loss, num_t = _bt_loss(model, data)
        step_losses.append(loss.item()); num_targets.append(num_t)
    return None, np.average(step_losses, weights=num_targets)

if __name__ == "__main__":
    skip_pf = os.environ.get("EXP15_SKIP_PREFLIGHT", "0") == "1"
    pf_only = os.environ.get("EXP15_PREFLIGHT_ONLY", "0") == "1"
    print(f"\n  EXP 15: MCMT-JEPA + Barlow Twins Redundancy Reduction + Cosine LR Warmup")
    print(f"  λ_BT={BT_LAMBDA}  proj_dim={BT_PROJ_DIM}  num_contexts={NUM_CONTEXTS}"
          f"  warmup={LR_WARMUP_EPOCHS}ep")

    if not skip_pf:
        try: run_preflight_all()
        except Exception as e: print(f"\n  PREFLIGHT FAILED: {e!r}"); raise
        if pf_only: sys.exit(0)

    wall_times = {}; tracker_results = {}; total_start = time.time()
    for dataset_name in DATASETS_TO_RUN:
        ds_start = time.time()
        print(f"\n{'#'*70}\n#  EXP 15 — BarlowTwins — DATASET: {dataset_name}\n{'#'*70}")
        updated_cfg = _merge_cfg_for_dataset(dataset_name)
        if dataset_name == "ZINC":
            tracker_results[dataset_name] = run(updated_cfg, create_dataset, create_model_bt, train, test)
        else:
            tracker_results[dataset_name] = run_k_fold(updated_cfg, create_dataset, create_model_bt, train, test)
        ds_elapsed = time.time() - ds_start
        wall_times[dataset_name] = ds_elapsed
        print(f"\n  [{dataset_name}] wall time: {ds_elapsed/60:.1f} min")

    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}\n  EXP 15 COMPLETE | Total: {total_elapsed/60:.1f} min\n{'='*70}")
    for ds, t in wall_times.items(): print(f"  {ds:<12}  {t/60:>9.1f}m")
    print_exp_tracker_footer(15, "MCMT-JEPA + Barlow Twins + Cosine LR Warmup", tracker_results)
