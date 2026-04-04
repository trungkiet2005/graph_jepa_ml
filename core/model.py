import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch_scatter import scatter
from einops.layers.torch import Rearrange
import core.model_utils.gMHA_wrapper as gMHA_wrapper

from core.model_utils.elements import MLP
from core.model_utils.feature_encoder import FeatureEncoder
from core.model_utils.gnn import GNN

class GraphJepa(nn.Module):

    def __init__(self,
                 nfeat_node, nfeat_edge,
                 nhid, nout,
                 nlayer_gnn,
                 nlayer_mlpmixer,
                 node_type, edge_type,
                 gnn_type,
                 gMHA_type='MLPMixer',
                 rw_dim=0,
                 lap_dim=0,
                 dropout=0,
                 mlpmixer_dropout=0,
                 bn=True,
                 res=True,
                 pooling='mean',
                 n_patches=32,
                 patch_rw_dim=0,
                 num_context_patches=1,
                 num_target_patches=4):

        super().__init__()
        self.dropout = dropout
        self.use_rw = rw_dim > 0
        self.use_lap = lap_dim > 0
        self.n_patches = n_patches
        self.pooling = pooling
        self.res = res
        self.patch_rw_dim = patch_rw_dim
        self.nhid = nhid
        self.nfeat_edge = nfeat_edge
        self.num_context_patches=num_context_patches
        self.num_target_patches=num_target_patches

        if self.use_rw:
            self.rw_encoder = MLP(rw_dim, nhid, 1)
        if self.use_lap:
            self.lap_encoder = MLP(lap_dim, nhid, 1)
        if self.patch_rw_dim > 0:
            self.patch_rw_encoder = MLP(self.patch_rw_dim, nhid, 1)

        self.input_encoder = FeatureEncoder(node_type, nfeat_node, nhid)
        self.edge_encoder = FeatureEncoder(edge_type, nfeat_edge, nhid)

        self.gnns = nn.ModuleList([GNN(nin=nhid, nout=nhid, nlayer_gnn=1, gnn_type=gnn_type,
                                  bn=bn, dropout=dropout, res=res) for _ in range(nlayer_gnn)])
        self.U = nn.ModuleList(
            [MLP(nhid, nhid, nlayer=1, with_final_activation=True) for _ in range(nlayer_gnn-1)])

        self.reshape = Rearrange('(B p) d ->  B p d', p=n_patches)

        self.context_encoder = getattr(gMHA_wrapper, 'Standard')(
            nhid=nhid, dropout=mlpmixer_dropout, nlayer=nlayer_mlpmixer, n_patches=n_patches)
        self.target_encoder = getattr(gMHA_wrapper, gMHA_type)(
            nhid=nhid, dropout=mlpmixer_dropout, nlayer=nlayer_mlpmixer, n_patches=n_patches)

        # Predictor
        self.target_predictor = MLP(
            nhid, 2, nlayer=3, with_final_activation=False, with_norm=False)

        # Use this predictor if you wish to do euclidean or poincaré embeddings in the latent space
        # self.target_predictor = MLP(
        #     nhid, 2, nlayer=3, with_final_activation=False, with_norm=False)

    def forward(self, data):
        x = self.input_encoder(data.x).squeeze()

        edge_attr = data.edge_attr
        if edge_attr is None:
            edge_attr = data.edge_index.new_zeros(data.edge_index.size(-1)).float().unsqueeze(-1)
        edge_attr = self.edge_encoder(edge_attr)

        # Patch Encoder
        x = x[data.subgraphs_nodes_mapper]
        edge_index = data.combined_subgraphs
        e = edge_attr[data.subgraphs_edges_mapper]
        batch_x = data.subgraphs_batch
        pes = data.rw_pos_enc[data.subgraphs_nodes_mapper]
        patch_pes = scatter(pes, batch_x, dim=0, reduce='max')
        for i, gnn in enumerate(self.gnns):
            if i > 0:
                subgraph = scatter(x, batch_x, dim=0,
                                   reduce=self.pooling)[batch_x]
                subgraph = scatter(x, batch_x, dim=0,
                                   reduce=self.pooling)[batch_x]
                x = x + self.U[i-1](subgraph)
                x = scatter(x, data.subgraphs_nodes_mapper,
                            dim=0, reduce='mean')[data.subgraphs_nodes_mapper]
            x = gnn(x, edge_index, e)
        subgraph_x = scatter(x, batch_x, dim=0, reduce=self.pooling)


        ######################## Graph-JEPA ########################
        # Create the correct indexer for each subgraph given the batching procedure
        batch_indexer = torch.tensor(np.cumsum(data.call_n_patches))
        batch_indexer = torch.hstack((torch.tensor(0), batch_indexer[:-1])).to(data.y.device)

        # Get idx of context and target subgraphs according to masks
        context_subgraph_idx = data.context_subgraph_idx + batch_indexer
        target_subgraphs_idx = torch.vstack([torch.tensor(dt) for dt in data.target_subgraph_idxs]).to(data.y.device)
        target_subgraphs_idx += batch_indexer.unsqueeze(1)

        # Get context and target subgraph (mpnn) embeddings
        context_subgraphs = subgraph_x[context_subgraph_idx]
        target_subgraphs = subgraph_x[target_subgraphs_idx.flatten()]

        # Construct context and target PEs frome the node pes of each subgraph
        target_pes = patch_pes[target_subgraphs_idx.flatten()]
        context_pe = patch_pes[context_subgraph_idx] 
        context_subgraphs += self.patch_rw_encoder(context_pe)
        encoded_tpatch_pes = self.patch_rw_encoder(target_pes)
        
        # Prepare inputs for MHA
        target_x = target_subgraphs.reshape(-1, self.num_target_patches, self.nhid)
        context_x = context_subgraphs.unsqueeze(1)

        # Given that there's only one element the attention operation "won't do anything"
        # This is simply for commodity of the EMA between context and target encoders
        context_mask = data.mask.flatten()[context_subgraph_idx].reshape(-1, self.num_context_patches) # this should be -1 x num context
        context_x = self.context_encoder(context_x, data.coarsen_adj if hasattr(
            data, 'coarsen_adj') else None, ~context_mask)

        # The target forward step musn't store gradients, since the target encoder is optimized via EMA
        with torch.no_grad():
            if hasattr(data, 'coarsen_adj'):
                subgraph_incides = torch.vstack([torch.tensor(dt) for dt in data.target_subgraph_idxs])
                patch_adj = data.coarsen_adj[
                    torch.arange(target_x.shape[0]).unsqueeze(1).unsqueeze(2),  # Batch dimension
                    subgraph_incides.unsqueeze(1),  # Row dimension
                    subgraph_incides.unsqueeze(2)   # Column dimension
                ]
                target_x = self.target_encoder(target_x, patch_adj, None)
            else:
                target_x = self.target_encoder(target_x, None, None)

            # Predict the coordinates of the patches in the Q1 hyperbola
            # Remove this part if you wish to do euclidean or poincaré embeddings in the latent space
            x_coord = torch.cosh(target_x.mean(-1).unsqueeze(-1))
            y_coord = torch.sinh(target_x.mean(-1).unsqueeze(-1))
            target_x = torch.cat([x_coord, y_coord], dim=-1)


        # Make predictions using the target predictor: for each target subgraph, we use the context + the target PE
        target_prediction_embeddings = context_x + encoded_tpatch_pes.reshape(-1, self.num_target_patches, self.nhid)
        target_y = self.target_predictor(target_prediction_embeddings)
       
        return target_x, target_y

    def encode(self, data):
        x = self.input_encoder(data.x).squeeze()

        edge_attr = data.edge_attr
        if edge_attr is None:
            edge_attr = data.edge_index.new_zeros(data.edge_index.size(-1)).float().unsqueeze(-1)
        edge_attr = self.edge_encoder(edge_attr)

        # Patch Encoder
        x = x[data.subgraphs_nodes_mapper]
        edge_index = data.combined_subgraphs
        e = edge_attr[data.subgraphs_edges_mapper]
        batch_x = data.subgraphs_batch
        pes = data.rw_pos_enc[data.subgraphs_nodes_mapper]
        patch_pes = scatter(pes, batch_x, dim=0, reduce='mean')
        for i, gnn in enumerate(self.gnns):
            if i > 0:
                subgraph = scatter(x, batch_x, dim=0,
                                   reduce=self.pooling)[batch_x]
                subgraph = scatter(x, batch_x, dim=0,
                                   reduce=self.pooling)[batch_x]
                x = x + self.U[i-1](subgraph)
                x = scatter(x, data.subgraphs_nodes_mapper,
                            dim=0, reduce='mean')[data.subgraphs_nodes_mapper]
            x = gnn(x, edge_index, e)
        subgraph_x = scatter(x, batch_x, dim=0, reduce=self.pooling)
        subgraph_x += self.patch_rw_encoder(patch_pes)
        
        # Handles different patch sizes based on the data object for multiscale training
        mixer_x = subgraph_x.reshape(len(data.call_n_patches), data.call_n_patches[0][0], -1)

        # Eval via target encoder
        mixer_x = self.target_encoder(mixer_x, data.coarsen_adj if hasattr(
                                        data, 'coarsen_adj') else None, ~data.mask) # Don't attend to empty patches when doing the final encoding
        
        # Global Average Pooling
        out = (mixer_x * data.mask.unsqueeze(-1)).sum(1) / data.mask.sum(1, keepdim=True)
        return out

    # ------------------------------------------------------------------

    def encode_nopool(self, data):
        x = self.input_encoder(data.x).squeeze()

        edge_attr = data.edge_attr
        if edge_attr is None:
            edge_attr = data.edge_index.new_zeros(data.edge_index.size(-1)).float().unsqueeze(-1)
        edge_attr = self.edge_encoder(edge_attr)

        # Patch Encoder
        x = x[data.subgraphs_nodes_mapper]
        edge_index = data.combined_subgraphs
        e = edge_attr[data.subgraphs_edges_mapper]
        batch_x = data.subgraphs_batch
        pes = data.rw_pos_enc[data.subgraphs_nodes_mapper]
        patch_pes = scatter(pes, batch_x, dim=0, reduce='mean')
        for i, gnn in enumerate(self.gnns):
            if i > 0:
                subgraph = scatter(x, batch_x, dim=0,
                                   reduce=self.pooling)[batch_x]
                subgraph = scatter(x, batch_x, dim=0,
                                   reduce=self.pooling)[batch_x]
                x = x + self.U[i-1](subgraph)
                x = scatter(x, data.subgraphs_nodes_mapper,
                            dim=0, reduce='mean')[data.subgraphs_nodes_mapper]
            x = gnn(x, edge_index, e)
        subgraph_x = scatter(x, batch_x, dim=0, reduce=self.pooling)
        subgraph_x += self.patch_rw_encoder(patch_pes)
        

        # Eval via target encoder
        mixer_x = subgraph_x.reshape(len(data.call_n_patches), data.call_n_patches[0], -1)
        mixer_x = self.target_encoder(mixer_x, data.coarsen_adj if hasattr(
                                        data, 'coarsen_adj') else None, ~data.mask) # Don't attend to empty patches when doing the final encoding

        return mixer_x


# ===========================================================================
# HMS-JEPA: Hierarchical Multi-Scale Graph JEPA
# ===========================================================================

class GraphHMSJepa(nn.Module):
    """Graph JEPA with 3-level hierarchical predictive coding.

    Three JEPA objectives are computed simultaneously:
      1. Same-scale  (L0 → L0): fine context predicts fine targets
      2. Cross-scale (L0 → L1): fine context predicts medium targets
      3. Cross-scale (L1 → L2): medium context predicts coarse targets

    The GNN backbone is *shared* across all scales; higher-level patch
    embeddings are obtained by mean-pooling the fine-level GNN output
    using the hierarchy indices produced by GraphHMSJEPAPartitionTransform.
    """

    def __init__(self,
                 nfeat_node, nfeat_edge,
                 nhid, nout,
                 nlayer_gnn,
                 nlayer_mlpmixer,
                 node_type, edge_type,
                 gnn_type,
                 gMHA_type='Hadamard',
                 rw_dim=0,
                 lap_dim=0,
                 dropout=0,
                 mlpmixer_dropout=0,
                 bn=True,
                 res=True,
                 pooling='mean',
                 n_patches=32,
                 patch_rw_dim=0,
                 num_context_patches=1,
                 num_target_patches=4,
                 num_target_patches_L1=4,
                 num_target_patches_L2=1,
                 loss_weights=(1.0, 0.5, 0.25),
                 var_weight=0.01):

        super().__init__()
        self.dropout = dropout
        self.use_rw = rw_dim > 0
        self.use_lap = lap_dim > 0
        self.n_patches = n_patches
        self.pooling = pooling
        self.res = res
        self.patch_rw_dim = patch_rw_dim
        self.nhid = nhid
        self.nfeat_edge = nfeat_edge
        self.num_context_patches = num_context_patches
        self.num_target_patches = num_target_patches
        self.num_target_patches_L1 = num_target_patches_L1
        self.num_target_patches_L2 = num_target_patches_L2
        self.loss_weights = list(loss_weights)
        self.var_weight = var_weight
        self.n_scales = 3

        if self.use_rw:
            self.rw_encoder = MLP(rw_dim, nhid, 1)
        if self.use_lap:
            self.lap_encoder = MLP(lap_dim, nhid, 1)

        # Patch positional encoders – one per scale
        if patch_rw_dim > 0:
            self.patch_rw_encoder    = MLP(patch_rw_dim, nhid, 1)
            self.patch_rw_encoder_L1 = MLP(patch_rw_dim, nhid, 1)
            self.patch_rw_encoder_L2 = MLP(patch_rw_dim, nhid, 1)

        self.input_encoder = FeatureEncoder(node_type, nfeat_node, nhid)
        self.edge_encoder  = FeatureEncoder(edge_type, nfeat_edge, nhid)

        # Shared GNN backbone (same as original GraphJepa)
        self.gnns = nn.ModuleList([GNN(nin=nhid, nout=nhid, nlayer_gnn=1, gnn_type=gnn_type,
                                       bn=bn, dropout=dropout, res=res) for _ in range(nlayer_gnn)])
        self.U = nn.ModuleList(
            [MLP(nhid, nhid, nlayer=1, with_final_activation=True) for _ in range(nlayer_gnn - 1)])

        # Three pairs of (context_encoder, target_encoder)
        # L0 – same as original
        self.context_encoder_L0 = getattr(gMHA_wrapper, 'Standard')(
            nhid=nhid, dropout=mlpmixer_dropout, nlayer=nlayer_mlpmixer, n_patches=n_patches)
        self.target_encoder_L0  = getattr(gMHA_wrapper, gMHA_type)(
            nhid=nhid, dropout=mlpmixer_dropout, nlayer=nlayer_mlpmixer, n_patches=n_patches)

        # L1 – always Hadamard (sequence-length-agnostic)
        self.context_encoder_L1 = getattr(gMHA_wrapper, 'Standard')(
            nhid=nhid, dropout=mlpmixer_dropout, nlayer=nlayer_mlpmixer, n_patches=n_patches)
        self.target_encoder_L1  = getattr(gMHA_wrapper, 'Hadamard')(
            nhid=nhid, dropout=mlpmixer_dropout, nlayer=nlayer_mlpmixer, n_patches=n_patches)

        # L2 – same
        self.context_encoder_L2 = getattr(gMHA_wrapper, 'Standard')(
            nhid=nhid, dropout=mlpmixer_dropout, nlayer=nlayer_mlpmixer, n_patches=n_patches)
        self.target_encoder_L2  = getattr(gMHA_wrapper, 'Hadamard')(
            nhid=nhid, dropout=mlpmixer_dropout, nlayer=nlayer_mlpmixer, n_patches=n_patches)

        # Convenience list for EMA updates in the training loop
        self.encoder_pairs = [
            (self.context_encoder_L0, self.target_encoder_L0),
            (self.context_encoder_L1, self.target_encoder_L1),
            (self.context_encoder_L2, self.target_encoder_L2),
        ]

        # Full-dimensional predictors (no 2-D hyperbolic bottleneck)
        self.predictor_L0_to_L0 = MLP(nhid, nhid, nlayer=3, with_final_activation=False, with_norm=False)
        self.predictor_L0_to_L1 = MLP(nhid, nhid, nlayer=2, with_final_activation=False, with_norm=False)
        self.predictor_L1_to_L2 = MLP(nhid, nhid, nlayer=2, with_final_activation=False, with_norm=False)

    # ------------------------------------------------------------------
    # Shared GNN forward
    # ------------------------------------------------------------------

    def _gnn_forward(self, data):
        """Run node feature encoding + GNN; return (subgraph_x_L0, raw_patch_pes, batch_x)."""
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

        for i, gnn in enumerate(self.gnns):
            if i > 0:
                subgraph = scatter(x, batch_x, dim=0, reduce=self.pooling)[batch_x]
                subgraph = scatter(x, batch_x, dim=0, reduce=self.pooling)[batch_x]
                x = x + self.U[i - 1](subgraph)
                x = scatter(x, data.subgraphs_nodes_mapper, dim=0, reduce='mean')[data.subgraphs_nodes_mapper]
            x = gnn(x, edge_index, e)

        subgraph_x_L0 = scatter(x, batch_x, dim=0, reduce=self.pooling)
        return subgraph_x_L0, raw_patch_pes, batch_x

    # ------------------------------------------------------------------
    # Forward (training)
    # ------------------------------------------------------------------

    def forward(self, data):
        device = data.y.device
        subgraph_x_L0, raw_patch_pes, _ = self._gnn_forward(data)

        # Hierarchical pooling (zero-copy: reuse L0 GNN output)
        subgraph_x_L1    = scatter(subgraph_x_L0, data.fine_to_medium,   dim=0, reduce='mean')
        raw_patch_pes_L1 = scatter(raw_patch_pes,  data.fine_to_medium,   dim=0, reduce='mean')
        subgraph_x_L2    = scatter(subgraph_x_L1,  data.medium_to_coarse, dim=0, reduce='mean')
        raw_patch_pes_L2 = scatter(raw_patch_pes_L1, data.medium_to_coarse, dim=0, reduce='mean')

        B = len(data.call_n_patches)

        def _batch_indexer(n_patches_list):
            bi = torch.tensor(np.cumsum(n_patches_list))
            return torch.hstack((torch.tensor(0), bi[:-1])).to(device)

        bi_L0 = _batch_indexer(data.call_n_patches)
        bi_L1 = _batch_indexer(data.n_patches_L1)
        bi_L2 = _batch_indexer(data.n_patches_L2)

        # ----------------------------------------------------------------
        # L0 same-scale JEPA  (context_L0 → predict target_L0)
        # ----------------------------------------------------------------
        ctx_idx_L0 = data.context_subgraph_idx + bi_L0
        tgt_idx_L0 = torch.vstack([torch.tensor(dt) for dt in data.target_subgraph_idxs]).to(device)
        tgt_idx_L0 += bi_L0.unsqueeze(1)

        ctx_patch_L0 = subgraph_x_L0[ctx_idx_L0]                               # [B, nhid]
        tgt_patch_L0 = subgraph_x_L0[tgt_idx_L0.flatten()]                     # [B*nT, nhid]
        ctx_pe_L0    = raw_patch_pes[ctx_idx_L0]
        tgt_pe_L0    = raw_patch_pes[tgt_idx_L0.flatten()]

        ctx_patch_L0        = ctx_patch_L0 + self.patch_rw_encoder(ctx_pe_L0)
        encoded_tgt_pe_L0   = self.patch_rw_encoder(tgt_pe_L0)

        ctx_x_L0  = ctx_patch_L0.unsqueeze(1)                                   # [B, 1, nhid]
        tgt_x_L0  = tgt_patch_L0.reshape(B, self.num_target_patches, self.nhid) # [B, nT, nhid]

        ctx_mask_L0 = data.mask.flatten()[ctx_idx_L0].reshape(B, self.num_context_patches)
        ctx_x_L0 = self.context_encoder_L0(ctx_x_L0,
                                            data.coarsen_adj if hasattr(data, 'coarsen_adj') else None,
                                            ~ctx_mask_L0)

        with torch.no_grad():
            if hasattr(data, 'coarsen_adj'):
                tgt_rel_L0 = torch.vstack([torch.tensor(dt) for dt in data.target_subgraph_idxs])
                patch_adj_L0 = data.coarsen_adj[
                    torch.arange(B).unsqueeze(1).unsqueeze(2),
                    tgt_rel_L0.unsqueeze(1),
                    tgt_rel_L0.unsqueeze(2)]
                tgt_x_L0 = self.target_encoder_L0(tgt_x_L0, patch_adj_L0, None)
            else:
                tgt_x_L0 = self.target_encoder_L0(tgt_x_L0, None, None)

        pred_L0 = self.predictor_L0_to_L0(
            ctx_x_L0 + encoded_tgt_pe_L0.reshape(B, self.num_target_patches, self.nhid))

        # ----------------------------------------------------------------
        # L0 → L1 cross-scale JEPA  (context_L0 predicts medium targets)
        # ----------------------------------------------------------------
        tgt_idx_L1 = torch.vstack([torch.tensor(dt) for dt in data.target_subgraph_idxs_L1]).to(device)
        tgt_idx_L1 += bi_L1.unsqueeze(1)
        nT_L1 = tgt_idx_L1.shape[1]

        tgt_patch_L1 = subgraph_x_L1[tgt_idx_L1.flatten()]                     # [B*nT_L1, nhid]
        tgt_pe_L1    = raw_patch_pes_L1[tgt_idx_L1.flatten()]
        encoded_tgt_pe_L1 = self.patch_rw_encoder_L1(tgt_pe_L1)

        tgt_x_L1 = tgt_patch_L1.reshape(B, nT_L1, self.nhid)

        with torch.no_grad():
            tgt_rel_L1 = torch.vstack([torch.tensor(dt) for dt in data.target_subgraph_idxs_L1])
            patch_adj_L1 = data.coarsen_adj_L1[
                torch.arange(B).unsqueeze(1).unsqueeze(2),
                tgt_rel_L1.unsqueeze(1),
                tgt_rel_L1.unsqueeze(2)]
            tgt_x_L1 = self.target_encoder_L1(tgt_x_L1, patch_adj_L1, None)

        pred_L1 = self.predictor_L0_to_L1(
            ctx_x_L0 + encoded_tgt_pe_L1.reshape(B, nT_L1, self.nhid))

        # ----------------------------------------------------------------
        # L1 → L2 cross-scale JEPA  (context_L1 predicts coarse targets)
        # ----------------------------------------------------------------
        ctx_idx_L1 = data.context_subgraph_idx_L1 + bi_L1
        ctx_patch_L1 = subgraph_x_L1[ctx_idx_L1]                               # [B, nhid]
        ctx_pe_L1    = raw_patch_pes_L1[ctx_idx_L1]
        ctx_patch_L1 = ctx_patch_L1 + self.patch_rw_encoder_L1(ctx_pe_L1)

        ctx_x_L1 = ctx_patch_L1.unsqueeze(1)                                   # [B, 1, nhid]
        ctx_mask_L1 = data.mask_L1.flatten()[ctx_idx_L1].reshape(B, 1)
        ctx_x_L1 = self.context_encoder_L1(ctx_x_L1, None, ~ctx_mask_L1)

        tgt_idx_L2 = torch.vstack([torch.tensor(dt) for dt in data.target_subgraph_idxs_L2]).to(device)
        tgt_idx_L2 += bi_L2.unsqueeze(1)
        nT_L2 = tgt_idx_L2.shape[1]

        tgt_patch_L2 = subgraph_x_L2[tgt_idx_L2.flatten()]                     # [B*nT_L2, nhid]
        tgt_pe_L2    = raw_patch_pes_L2[tgt_idx_L2.flatten()]
        encoded_tgt_pe_L2 = self.patch_rw_encoder_L2(tgt_pe_L2)

        tgt_x_L2 = tgt_patch_L2.reshape(B, nT_L2, self.nhid)

        with torch.no_grad():
            tgt_rel_L2 = torch.vstack([torch.tensor(dt) for dt in data.target_subgraph_idxs_L2])
            patch_adj_L2 = data.coarsen_adj_L2[
                torch.arange(B).unsqueeze(1).unsqueeze(2),
                tgt_rel_L2.unsqueeze(1),
                tgt_rel_L2.unsqueeze(2)]
            tgt_x_L2 = self.target_encoder_L2(tgt_x_L2, patch_adj_L2, None)

        pred_L2 = self.predictor_L1_to_L2(
            ctx_x_L1 + encoded_tgt_pe_L2.reshape(B, nT_L2, self.nhid))

        return (tgt_x_L0, pred_L0), (tgt_x_L1, pred_L1), (tgt_x_L2, pred_L2)

    # ------------------------------------------------------------------
    # Encode (inference)
    # ------------------------------------------------------------------

    def encode(self, data):
        subgraph_x_L0, raw_patch_pes, _ = self._gnn_forward(data)

        subgraph_x_L1    = scatter(subgraph_x_L0, data.fine_to_medium,    dim=0, reduce='mean')
        raw_patch_pes_L1 = scatter(raw_patch_pes,  data.fine_to_medium,    dim=0, reduce='mean')
        subgraph_x_L2    = scatter(subgraph_x_L1,  data.medium_to_coarse,  dim=0, reduce='mean')
        raw_patch_pes_L2 = scatter(raw_patch_pes_L1, data.medium_to_coarse, dim=0, reduce='mean')

        # Add patch PEs
        subgraph_x_L0 = subgraph_x_L0 + self.patch_rw_encoder(raw_patch_pes)
        subgraph_x_L1 = subgraph_x_L1 + self.patch_rw_encoder_L1(raw_patch_pes_L1)
        subgraph_x_L2 = subgraph_x_L2 + self.patch_rw_encoder_L2(raw_patch_pes_L2)

        B    = len(data.call_n_patches)
        n_L0 = data.call_n_patches[0][0]
        n_L1 = data.n_patches_L1[0][0]
        n_L2 = data.n_patches_L2[0][0]

        mixer_L0 = subgraph_x_L0.reshape(B, n_L0, -1)
        mixer_L1 = subgraph_x_L1.reshape(B, n_L1, -1)
        mixer_L2 = subgraph_x_L2.reshape(B, n_L2, -1)

        mixer_L0 = self.target_encoder_L0(mixer_L0,
                                           data.coarsen_adj if hasattr(data, 'coarsen_adj') else None,
                                           ~data.mask)
        mixer_L1 = self.target_encoder_L1(mixer_L1, data.coarsen_adj_L1, ~data.mask_L1)
        mixer_L2 = self.target_encoder_L2(mixer_L2, data.coarsen_adj_L2, ~data.mask_L2)

        # Masked average pooling at L0; plain mean at L1/L2 (all patches filled)
        out_L0 = (mixer_L0 * data.mask.unsqueeze(-1)).sum(1) / data.mask.sum(1, keepdim=True)
        out_L1 = mixer_L1.mean(1)
        out_L2 = mixer_L2.mean(1)

        return torch.cat([out_L0, out_L1, out_L2], dim=-1)    # [B, 3*nhid]