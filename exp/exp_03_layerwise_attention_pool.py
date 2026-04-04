# ============================================================
# EXP 03 — Layer-Wise Attention Pooling (HISTOGRAPH-inspired)
# ============================================================
# Papers: "HISTOGRAPH: Learning from Historical Activations" (2026)
#         "DeeperGCN: All You Need to Train Deeper GCNs" (NeurIPS 2020)
#
# Idea:   The baseline HMS-JEPA only uses the LAST GNN layer output
#         to produce subgraph embeddings. But intermediate GNN layers
#         capture different structural information:
#           - Early layers: local neighborhood (1-2 hops)
#           - Later layers: broader context
#
#         We add a learnable attention mechanism that pools across
#         ALL GNN layer outputs (not just the last one) to produce
#         richer subgraph embeddings. This is a "free" improvement
#         since the intermediate features are already computed.
#
# Key changes:
#   - Store all intermediate GNN layer outputs
#   - Learnable attention weights per layer (LayerAttentionPool)
#   - Weighted sum produces the final subgraph embedding
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
from core.model_utils.feature_encoder import FeatureEncoder
from core.model_utils.gnn import GNN
import core.model_utils.gMHA_wrapper as gMHA_wrapper
from train.zinc     import _compute_loss, _ema_update

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

_cfg_path = "/tmp/exp03_layerwise.yaml"
with open(_cfg_path, "w") as f:
    f.write(CONFIG)

# ─────────────────────────────────────────────────────────────
# 3. MODEL — GraphHMSJepa with Layer-Wise Attention Pooling
# ─────────────────────────────────────────────────────────────

class LayerAttentionPool(nn.Module):
    """Learnable attention-weighted pooling across GNN layers.

    Given features from L GNN layers [x_1, ..., x_L], each [N, D],
    compute attention weights α_l and return sum(α_l * x_l).
    """

    def __init__(self, nhid, nlayers):
        super().__init__()
        # One attention score per layer, context-dependent
        self.attn = nn.Sequential(
            nn.Linear(nhid, nhid // 4),
            nn.ReLU(),
            nn.Linear(nhid // 4, 1),
        )
        self.nlayers = nlayers

    def forward(self, layer_outputs):
        """
        Args:
            layer_outputs: list of [N, D] tensors, one per GNN layer
        Returns:
            [N, D] attention-weighted combination
        """
        # Stack: [N, L, D]
        stacked = torch.stack(layer_outputs, dim=1)
        # Attention scores: [N, L, 1]
        scores = self.attn(stacked)
        # Normalize across layers
        weights = F.softmax(scores, dim=1)
        # Weighted sum: [N, D]
        return (weights * stacked).sum(dim=1)


class GraphHMSJepaLayerAttn(GraphHMSJepa):
    """HMS-JEPA with layer-wise attention pooling over GNN outputs.

    Instead of using only the last GNN layer's output, we collect
    outputs from ALL GNN layers and use learned attention to combine them.
    """

    def __init__(self, *args, nlayer_gnn=2, **kwargs):
        super().__init__(*args, nlayer_gnn=nlayer_gnn, **kwargs)
        # Add layer attention pooling
        self.layer_attn = LayerAttentionPool(self.nhid, nlayer_gnn)

    def _gnn_forward(self, data):
        """Override: collect all layer outputs and attention-pool them."""
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
            # Collect each layer's subgraph-level pooling
            layer_pool = scatter(x, batch_x, dim=0, reduce=self.pooling)
            layer_outputs.append(layer_pool)

        # Attention-weighted combination across layers
        subgraph_x_L0 = self.layer_attn(layer_outputs)

        return subgraph_x_L0, raw_patch_pes, batch_x


def create_model_layerattn(cfg):
    """Create GraphHMSJepaLayerAttn instead of default model."""
    # Dataset-specific features (same as get_model.py)
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

    return GraphHMSJepaLayerAttn(
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
# 4. TRAIN / TEST  (standard — uses shared _compute_loss/_ema_update)
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

    run_k_fold(cfg, create_dataset, create_model_layerattn, train, test)
