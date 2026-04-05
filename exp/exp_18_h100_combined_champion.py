# ============================================================
# EXP 18 — Combined Champion + Extended Training + Full 8-Dataset Suite
# ============================================================
# Strategic combination of all winning elements from EXP 08-17:
#   - MCMT multi-context from EXP 13 (2 context patches)
#   - NT-Xent auxiliary loss from EXP 13
#   - Barlow Twins as secondary auxiliary (if EXP 15 improves)
#   - Light DropPath 0.03 globally
#   - Cosine LR warmup (5 epochs)
#   - EMA 0.99→1.0 cosine schedule
#   - Per-dataset HP tuning (critical for DD, REDDIT)
#   - Full 8-dataset suite: MUTAG, PROTEINS, IMDB-B/M, DD, ZINC,
#     REDDIT-B, REDDIT-M5
#
# Datasets: MUTAG, PROTEINS, IMDB-BINARY, IMDB-MULTI, DD, ZINC,
#           REDDIT-BINARY, REDDIT-MULTI5K
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

# ─────────────────────────────────────────────────────────────
# HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────
NUM_CONTEXTS        = 2
NT_XENT_WEIGHT      = 0.05
NT_XENT_TEMPERATURE = 0.1
DROPPATH_RATE       = 0.03   # light, global DropPath
EMA_START           = 0.99
EMA_END             = 1.0
LR_WARMUP_EPOCHS    = 5


# ─────────────────────────────────────────────────────────────
# DROP PATH (from EXP 10, simplified global rate)
# ─────────────────────────────────────────────────────────────
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep_prob, device=x.device, dtype=x.dtype))
        return x * mask / keep_prob


# ─────────────────────────────────────────────────────────────
# NT-XENT LOSS
# ─────────────────────────────────────────────────────────────
def nt_xent_loss(z1, z2, temperature=NT_XENT_TEMPERATURE):
    B = z1.shape[0]
    z1 = F.normalize(z1, dim=-1); z2 = F.normalize(z2, dim=-1)
    z  = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.T) / temperature
    mask = torch.eye(2*B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float('-inf'))
    labels = torch.cat([torch.arange(B, 2*B), torch.arange(B)]).to(z.device)
    return F.cross_entropy(sim, labels)


# ─────────────────────────────────────────────────────────────
# MULTI-CONTEXT CROSS-ATTENTION MODULE
# ─────────────────────────────────────────────────────────────
class MultiContextAggregator(nn.Module):
    def __init__(self, nhid: int, nhead: int = 8, drop_path: float = 0.0):
        super().__init__()
        self.ctx_self_attn = nn.MultiheadAttention(nhid, nhead, batch_first=True)
        self.ctx_norm1     = nn.LayerNorm(nhid)
        self.cross_attn    = nn.MultiheadAttention(nhid, nhead, batch_first=True)
        self.cross_norm    = nn.LayerNorm(nhid)
        self.ffn  = nn.Sequential(nn.Linear(nhid, nhid*2), nn.GELU(), nn.Linear(nhid*2, nhid))
        self.norm2 = nn.LayerNorm(nhid)
        self.dp = DropPath(drop_path)

    def forward(self, ctx_x, tgt_pe):
        ctx_sa, _ = self.ctx_self_attn(ctx_x, ctx_x, ctx_x)
        ctx_x = self.ctx_norm1(ctx_x + self.dp(ctx_sa))
        ctx_ca, _ = self.cross_attn(tgt_pe, ctx_x, ctx_x)
        out = self.cross_norm(tgt_pe + self.dp(ctx_ca))
        return self.norm2(out + self.dp(self.ffn(out)))


# ─────────────────────────────────────────────────────────────
# MODEL: COMBINED CHAMPION
# ─────────────────────────────────────────────────────────────
class GraphHMSJepaChampion(GraphHMSJepa):
    """
    Combined Champion: MCMT + NT-Xent + DropPath + extended training.
    Best elements from EXP 08-17.
    """
    def __init__(self, *args, num_contexts=2, drop_path=DROPPATH_RATE, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_contexts = num_contexts
        nhid = self.nhid
        self.mc_agg_L0 = MultiContextAggregator(nhid, drop_path=drop_path)
        self.mc_agg_L1 = MultiContextAggregator(nhid, drop_path=drop_path)
        self.dp = DropPath(drop_path)
        self.proj_head = nn.Sequential(
            nn.Linear(nhid, nhid), nn.GELU(), nn.Linear(nhid, 128))

    def forward_champion(self, data):
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

        # L0 dual-context
        ctx_idx  = data.context_subgraph_idx + bi_L0
        ctx1 = subgraph_x_L0[ctx_idx] + self.patch_rw_encoder(raw_patch_pes[ctx_idx])
        n_ppg = torch.tensor([c[0] for c in data.call_n_patches], device=device)
        ctx2_idx = ((data.context_subgraph_idx + 1) % n_ppg) + bi_L0
        ctx2 = subgraph_x_L0[ctx2_idx] + self.patch_rw_encoder(raw_patch_pes[ctx2_idx])
        ctx_stack = torch.stack([ctx1, ctx2], dim=1)  # [B,2,H]

        # Target L0
        tgt_idx_L0 = torch.vstack([torch.tensor(dt) for dt in data.target_subgraph_idxs]).to(device) + bi_L0.unsqueeze(1)
        tgt_pe_L0 = self.patch_rw_encoder(raw_patch_pes[tgt_idx_L0.flatten()]).reshape(B, self.num_target_patches, self.nhid)
        tgt_x_L0 = subgraph_x_L0[tgt_idx_L0.flatten()].reshape(B, self.num_target_patches, self.nhid)
        with torch.no_grad():
            if hasattr(data, 'coarsen_adj'):
                rel = torch.vstack([torch.tensor(dt) for dt in data.target_subgraph_idxs])
                adj = data.coarsen_adj[torch.arange(B).unsqueeze(1).unsqueeze(2), rel.unsqueeze(1), rel.unsqueeze(2)]
                tgt_x_L0 = self.target_encoder_L0(tgt_x_L0, adj, None)
            else:
                tgt_x_L0 = self.target_encoder_L0(tgt_x_L0, None, None)
        pred_L0 = self.predictor_L0_to_L0(self.mc_agg_L0(ctx_stack, tgt_pe_L0))

        # NT-Xent views
        z1 = self.proj_head(self.context_encoder_L0(ctx1.unsqueeze(1), None, None).squeeze(1))
        z2 = self.proj_head(self.context_encoder_L0(ctx2.unsqueeze(1), None, None).squeeze(1))

        # L0→L1
        tgt_idx_L1 = torch.vstack([torch.tensor(dt) for dt in data.target_subgraph_idxs_L1]).to(device) + bi_L1.unsqueeze(1)
        nT_L1 = tgt_idx_L1.shape[1]
        tgt_x_L1 = subgraph_x_L1[tgt_idx_L1.flatten()].reshape(B, nT_L1, self.nhid)
        with torch.no_grad():
            rel_L1 = torch.vstack([torch.tensor(dt) for dt in data.target_subgraph_idxs_L1])
            adj_L1 = data.coarsen_adj_L1[torch.arange(B).unsqueeze(1).unsqueeze(2), rel_L1.unsqueeze(1), rel_L1.unsqueeze(2)]
            tgt_x_L1 = self.target_encoder_L1(tgt_x_L1, adj_L1, None)
        tgt_pe_L1 = self.patch_rw_encoder_L1(raw_patch_pes_L1[tgt_idx_L1.flatten()]).reshape(B, nT_L1, self.nhid)
        pred_L1 = self.predictor_L0_to_L1(self.mc_agg_L1(ctx_stack, tgt_pe_L1))

        # L1→L2
        ctx_idx_L1 = data.context_subgraph_idx_L1 + bi_L1
        ctx_L1 = subgraph_x_L1[ctx_idx_L1] + self.patch_rw_encoder_L1(raw_patch_pes_L1[ctx_idx_L1])
        ctx_x_L1 = self.context_encoder_L1(ctx_L1.unsqueeze(1), None, ~data.mask_L1.flatten()[ctx_idx_L1].reshape(B,1))
        tgt_idx_L2 = torch.vstack([torch.tensor(dt) for dt in data.target_subgraph_idxs_L2]).to(device) + bi_L2.unsqueeze(1)
        nT_L2 = tgt_idx_L2.shape[1]
        tgt_x_L2 = subgraph_x_L2[tgt_idx_L2.flatten()].reshape(B, nT_L2, self.nhid)
        with torch.no_grad():
            rel_L2 = torch.vstack([torch.tensor(dt) for dt in data.target_subgraph_idxs_L2])
            adj_L2 = data.coarsen_adj_L2[torch.arange(B).unsqueeze(1).unsqueeze(2), rel_L2.unsqueeze(1), rel_L2.unsqueeze(2)]
            tgt_x_L2 = self.target_encoder_L2(tgt_x_L2, adj_L2, None)
        pe_L2 = self.patch_rw_encoder_L2(raw_patch_pes_L2[tgt_idx_L2.flatten()]).reshape(B, nT_L2, self.nhid)
        pred_L2 = self.predictor_L1_to_L2(ctx_x_L1 + pe_L2)

        return (tgt_x_L0, pred_L0), (tgt_x_L1, pred_L1), (tgt_x_L2, pred_L2), (z1, z2)


# ─────────────────────────────────────────────────────────────
# LOSS
# ─────────────────────────────────────────────────────────────
def _champion_loss(model, data):
    if not isinstance(model, GraphHMSJepaChampion):
        from train.zinc import _compute_loss
        return _compute_loss(model, data, nn.SmoothL1Loss(beta=0.5), 0)
    criterion = nn.SmoothL1Loss(beta=0.5)
    (tx0,ty0),(tx1,ty1),(tx2,ty2),(z1,z2) = model.forward_champion(data)
    w = model.loss_weights
    loss = w[0]*criterion(ty0,tx0) + w[1]*criterion(ty1,tx1) + w[2]*criterion(ty2,tx2)
    loss = loss + model.var_weight * torch.mean(torch.relu(1.0 - tx0.detach().std(dim=0)))
    if z1.shape[0] > 1:
        loss = loss + NT_XENT_WEIGHT * nt_xent_loss(z1, z2)
    return loss, len(ty0)


# ─────────────────────────────────────────────────────────────
# MODEL FACTORY
# ─────────────────────────────────────────────────────────────
def _resolve_dataset_types(name):
    m = {'MUTAG':('Linear','Linear',7,4),'PROTEINS':('Linear','Linear',3,1),
         'DD':('Linear','Linear',89,1),'IMDB-BINARY':('Linear','Linear',1,1),
         'IMDB-MULTI':('Linear','Linear',1,1),'REDDIT-BINARY':('Linear','Linear',1,1),
         'REDDIT-MULTI5K':('Linear','Linear',1,1),
         'ZINC':('Discrete','Discrete',28,4)}
    return m.get(name, ('Linear','Linear',1,1))

def create_model_champion(cfg):
    base = create_model(cfg)
    if not isinstance(base, GraphHMSJepa): return base
    nt, et, nfn, nfe = _resolve_dataset_types(cfg.dataset)
    dp = DROPPATH_RATE
    # Per-dataset DropPath override
    if cfg.dataset == 'DD':
        dp = 0.02
    model = GraphHMSJepaChampion(
        nfeat_node=nfn, nfeat_edge=nfe, nhid=base.nhid, nout=1,
        nlayer_gnn=len(base.gnns), nlayer_mlpmixer=cfg.model.nlayer_mlpmixer,
        node_type=nt, edge_type=et, gnn_type=cfg.model.gnn_type, gMHA_type=cfg.model.gMHA_type,
        rw_dim=cfg.pos_enc.rw_dim, lap_dim=cfg.pos_enc.lap_dim,
        dropout=getattr(cfg.train,'dropout',0), mlpmixer_dropout=getattr(cfg.train,'mlpmixer_dropout',0),
        n_patches=cfg.metis.n_patches, patch_rw_dim=cfg.pos_enc.patch_rw_dim,
        num_context_patches=cfg.jepa.num_context, num_target_patches=cfg.jepa.num_targets,
        num_target_patches_L1=cfg.jepa.num_targets_L1, num_target_patches_L2=cfg.jepa.num_targets_L2,
        loss_weights=cfg.jepa.loss_weights, var_weight=cfg.jepa.var_weight,
        num_contexts=NUM_CONTEXTS, drop_path=dp,
    )
    p = sum(x.numel() for x in model.parameters() if x.requires_grad)
    print(f"  [Champion Model] params={p/1e6:.2f}M  dp={dp}  EMA={EMA_START}→{EMA_END}")
    return model


# ─────────────────────────────────────────────────────────────
# PER-DATASET CONFIGS (tuned: extended epochs, per-ds HP)
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
    "REDDIT-BINARY": """
dataset: REDDIT-BINARY
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
  batch_size: 256
  lr: 0.0005
  runs: 5
metis:
  n_patches: 128
  online: False
pos_enc:
  rw_dim: 40
  patch_rw_dim: 40
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
    "REDDIT-MULTI5K": """
dataset: REDDIT-MULTI5K
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
  batch_size: 256
  lr: 0.0005
  runs: 5
metis:
  n_patches: 128
  online: False
pos_enc:
  rw_dim: 40
  patch_rw_dim: 40
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

DATASETS_TO_RUN = [
    "MUTAG", "PROTEINS", "IMDB-BINARY", "IMDB-MULTI", "DD", "ZINC",
    "REDDIT-BINARY", "REDDIT-MULTI5K",
]
_PREFLIGHT_MAX_GRAPHS = 24

def _device_tensor(cfg):
    d = cfg.device
    if isinstance(d, torch.device): return d
    if isinstance(d, str): return torch.device(d)
    if isinstance(d, int): return torch.device(f"cuda:{d}" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")

def _merge_cfg_for_dataset(ds):
    cfg_path = f"/tmp/exp18_{ds.replace('-','_').lower()}.yaml"
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
        loader = DataLoader([x for x in sub], batch_size=min(cfg.train.batch_size,max(1,n)), shuffle=True, num_workers=0)
    model = create_model_champion(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    data = next(iter(loader)).to(device); optimizer.zero_grad()
    with torch.amp.autocast("cuda", enabled=USE_AMP, dtype=AMP_DTYPE):
        loss, _ = _champion_loss(model, data)
    scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()

def run_preflight_all():
    print("\n" + "="*70)
    print("  EXP 18 — PREFLIGHT (Combined Champion — 8 Datasets)")
    print("="*70)
    t0 = time.time()
    for name in DATASETS_TO_RUN:
        t_ds = time.time()
        print(f"\n  [preflight] {name} ...", flush=True)
        try:
            preflight_one_dataset(name)
            print(f"  [preflight] {name} OK ({time.time()-t_ds:.1f}s)", flush=True)
        except Exception as e:
            print(f"  [preflight] {name} FAILED: {e!r}", flush=True)
            print(f"  (will skip in training if preflight fails)")
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
            loss, num_t = _champion_loss(model, data)
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
            loss, num_t = _champion_loss(model, data)
        step_losses.append(loss.item()); num_targets.append(num_t)
    return None, np.average(step_losses, weights=num_targets)

if __name__ == "__main__":
    skip_pf = os.environ.get("EXP18_SKIP_PREFLIGHT", "0") == "1"
    pf_only = os.environ.get("EXP18_PREFLIGHT_ONLY", "0") == "1"
    print(f"\n  EXP 18: Combined Champion + Extended Training + 8 Datasets")
    print(f"  MCMT + NT-Xent + DropPath={DROPPATH_RATE} + EMA {EMA_START}→{EMA_END}")
    print(f"  Datasets: {', '.join(DATASETS_TO_RUN)}")

    if not skip_pf:
        try: run_preflight_all()
        except Exception as e: print(f"\n  PREFLIGHT FAILED: {e!r}"); raise
        if pf_only: sys.exit(0)

    wall_times = {}; tracker_results = {}; total_start = time.time()
    for dataset_name in DATASETS_TO_RUN:
        ds_start = time.time()
        print(f"\n{'#'*70}\n#  EXP 18 — Champion — DATASET: {dataset_name}\n{'#'*70}")
        try:
            updated_cfg = _merge_cfg_for_dataset(dataset_name)
            if dataset_name == "ZINC":
                tracker_results[dataset_name] = run(updated_cfg, create_dataset, create_model_champion, train, test)
            else:
                tracker_results[dataset_name] = run_k_fold(updated_cfg, create_dataset, create_model_champion, train, test)
        except Exception as e:
            print(f"\n  [{dataset_name}] FAILED: {e!r}")
            tracker_results[dataset_name] = None
        ds_elapsed = time.time() - ds_start
        wall_times[dataset_name] = ds_elapsed
        print(f"\n  [{dataset_name}] wall time: {ds_elapsed/60:.1f} min")

    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}\n  EXP 18 COMPLETE | Total: {total_elapsed/60:.1f} min\n{'='*70}")
    for ds, t in wall_times.items(): print(f"  {ds:<16}  {t/60:>9.1f}m")
    print_exp_tracker_footer(18, "Combined Champion + 8 Datasets", tracker_results)
