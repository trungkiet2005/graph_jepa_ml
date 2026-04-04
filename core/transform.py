import re
import random
import torch
import numpy as np
from torch_geometric.data import Data
from core.transform_utils.subgraph_extractors import metis_subgraph, random_subgraph
from core.data_utils.pe import RWSE, LapPE, random_walk
from torch_geometric.transforms import Compose

from torch_geometric.utils import degree
from torch_geometric.loader import NeighborLoader, GraphSAINTSampler

try:
    import metis as metis_lib
    import networkx as nx
    _METIS_AVAILABLE = True
except ImportError:
    _METIS_AVAILABLE = False

def cal_coarsen_adj(subgraphs_nodes_mask):
    mask = subgraphs_nodes_mask.to(torch.float)
    coarsen_adj = torch.matmul(mask, mask.t())
    return coarsen_adj


def to_sparse(node_mask, edge_mask):
    subgraphs_nodes = node_mask.nonzero().T
    subgraphs_edges = edge_mask.nonzero().T
    return subgraphs_nodes, subgraphs_edges


def combine_subgraphs(edge_index, subgraphs_nodes, subgraphs_edges, num_selected=None, num_nodes=None):
    if num_selected is None:
        num_selected = subgraphs_nodes[0][-1] + 1
    if num_nodes is None:
        num_nodes = subgraphs_nodes[1].max() + 1

    combined_subgraphs = edge_index[:, subgraphs_edges[1]]
    node_label_mapper = edge_index.new_full((num_selected, num_nodes), -1)
    node_label_mapper[subgraphs_nodes[0], subgraphs_nodes[1]
                      ] = torch.arange(len(subgraphs_nodes[1]))
    node_label_mapper = node_label_mapper.reshape(-1)

    inc = torch.arange(num_selected)*num_nodes
    combined_subgraphs += inc[subgraphs_edges[0]]
    combined_subgraphs = node_label_mapper[combined_subgraphs]
    return combined_subgraphs

def combine_subgraphs_jepa(edge_index, subgraphs_nodes, subgraphs_edges, 
                           context_nodes_mask, context_edges_mask, 
                           target_nodes_mask, target_edges_mask, 
                           num_selected=None, num_nodes=None):
    if num_selected is None:
        num_selected = subgraphs_nodes[0][-1] + 1
    if num_nodes is None:
        num_nodes = subgraphs_nodes[1].max() + 1

    combined_subgraphs = edge_index[:, subgraphs_edges[1]] # Select all the subgraph edges from the global edge index
    node_label_mapper = edge_index.new_full((num_selected, num_nodes), -1)
    node_label_mapper[subgraphs_nodes[0], subgraphs_nodes[1]
                      ] = torch.arange(len(subgraphs_nodes[1])) # For each subgraph, create the new "placeholder" indices
    node_label_mapper = node_label_mapper.reshape(-1)

    inc = torch.arange(num_selected)*num_nodes
    combined_subgraphs += inc[subgraphs_edges[0]]
    combined_subgraphs = node_label_mapper[combined_subgraphs]

    return combined_subgraphs


class SubgraphsData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        num_nodes = self.num_nodes
        num_edges = self.edge_index.size(-1)
        if bool(re.search('(combined_subgraphs)', key)):
            return getattr(self, key[:-len('combined_subgraphs')]+'subgraphs_nodes_mapper').size(0)
        elif bool(re.search('(subgraphs_batch)', key)):
            return 1+getattr(self, key)[-1]
        elif bool(re.search('(nodes_mapper)', key)):
            return num_nodes
        elif bool(re.search('(edges_mapper)', key)):
            return num_edges
        elif bool(re.search('(fine_to_medium)', key)):
            if hasattr(self, 'n_patches_L1'):
                v = self.n_patches_L1
                return v[0] if isinstance(v, list) else int(v)
            return 0
        elif bool(re.search('(medium_to_coarse)', key)):
            if hasattr(self, 'n_patches_L2'):
                v = self.n_patches_L2
                return v[0] if isinstance(v, list) else int(v)
            return 0
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if bool(re.search('(combined_subgraphs)', key)):
            return -1
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)


class PositionalEncodingTransform(object):
    def __init__(self, rw_dim=0, lap_dim=0):
        super().__init__()
        self.rw_dim = rw_dim
        self.lap_dim = lap_dim

    def __call__(self, data):
        if self.rw_dim > 0:
            data.rw_pos_enc = RWSE(
                data.edge_index, self.rw_dim, data.num_nodes)
        if self.lap_dim > 0:
            data.lap_pos_enc = LapPE(
                data.edge_index, self.lap_dim, data.num_nodes)
        return data


class GraphJEPAPartitionTransform(object):
    def __init__(self, n_patches, metis=True, drop_rate=0.0, num_hops=1, is_directed=False, patch_rw_dim=0, patch_num_diff=0, num_context=1, num_targets=4):
        super().__init__()
        self.n_patches = n_patches
        self.drop_rate = drop_rate
        self.num_hops = num_hops
        self.is_directed = is_directed
        self.patch_rw_dim = patch_rw_dim
        self.patch_num_diff = patch_num_diff
        self.metis = metis
        self.num_context = num_context
        self.num_targets = num_targets

    def _diffuse(self, A):
        if self.patch_num_diff == 0:
            return A
        Dinv = A.sum(dim=-1).clamp(min=1).pow(-1).unsqueeze(-1)  # D^-1
        RW = A * Dinv
        M = RW
        M_power = M
        # Iterate
        for _ in range(self.patch_num_diff-1):
            M_power = torch.matmul(M_power, M)
        return M_power

    def __call__(self, data):
        data = SubgraphsData(**{k: v for k, v in data})
        if self.metis:
            node_masks, edge_masks = metis_subgraph(
                data, n_patches=self.n_patches, drop_rate=self.drop_rate, num_hops=self.num_hops, is_directed=self.is_directed)
        else:
            node_masks, edge_masks = random_subgraph(
                data, n_patches=self.n_patches, num_hops=self.num_hops)
        subgraphs_nodes, subgraphs_edges = to_sparse(node_masks, edge_masks)

        
        combined_subgraphs = combine_subgraphs(
            data.edge_index, subgraphs_nodes, subgraphs_edges, 
            num_selected=self.n_patches, num_nodes=data.num_nodes)

        if self.patch_num_diff > -1 or self.patch_rw_dim > 0:
            coarsen_adj = cal_coarsen_adj(node_masks)
            if self.patch_rw_dim > 0:
                data.patch_pe = random_walk(coarsen_adj, self.patch_rw_dim)
            if self.patch_num_diff > -1:
                data.coarsen_adj = self._diffuse(coarsen_adj).unsqueeze(0)

        subgraphs_batch = subgraphs_nodes[0]
        mask = torch.zeros(self.n_patches).bool()
        mask[subgraphs_batch] = True
        data.subgraphs_batch = subgraphs_batch
        data.subgraphs_nodes_mapper = subgraphs_nodes[1]
        data.subgraphs_edges_mapper = subgraphs_edges[1]
        data.combined_subgraphs = combined_subgraphs
        data.mask = mask.unsqueeze(0)

        # Pick one subgraph as context and others as targets (at random)
        # Attention computation (mask) fix: Select only from non-empty patches
        rand_choice = np.random.choice(subgraphs_nodes[0].unique(), self.num_context+self.num_targets, replace=False)
        context_subgraph_idx = rand_choice[0]
        target_subgraph_idxs = torch.tensor(rand_choice[1:])

        data.context_edges_mask = subgraphs_edges[0] == context_subgraph_idx
        data.target_edges_mask = torch.isin(subgraphs_edges[0], target_subgraph_idxs)
        data.context_nodes_mapper = subgraphs_nodes[1, subgraphs_nodes[0] == context_subgraph_idx]
        data.target_nodes_mapper = subgraphs_nodes[1, torch.isin(subgraphs_nodes[0], target_subgraph_idxs)]
        data.context_nodes_subgraph = subgraphs_nodes[0, subgraphs_nodes[0] == context_subgraph_idx]
        data.target_nodes_subgraph = subgraphs_nodes[0, torch.isin(subgraphs_nodes[0], target_subgraph_idxs)]
        data.context_subgraph_idx = context_subgraph_idx.tolist()
        data.target_subgraph_idxs = target_subgraph_idxs.tolist()
        data.call_n_patches = [self.n_patches]

        data.__num_nodes__ = data.num_nodes  # set number of nodes of the current graph
        return data


class GraphHMSJEPAPartitionTransform(object):
    """Hierarchical Multi-Scale JEPA partition transform.

    Builds a 3-level patch hierarchy:
      L0 (fine)   : n_patches          patches  (default 32)
      L1 (medium) : n_patches_L1       patches  (default  8)
      L2 (coarse) : n_patches_L2       patches  (default  2)

    L1 and L2 are derived by running METIS on the coarsened patch-level
    adjacency produced at the previous level, so the hierarchy is
    *consistent*: every coarse patch is a union of medium patches.
    """

    def __init__(self, n_patches, scale_factor=4, metis_enable=True,
                 drop_rate=0.0, num_hops=1, is_directed=False,
                 patch_rw_dim=0, patch_num_diff=0,
                 num_context=1, num_targets=4,
                 num_targets_L1=4, num_targets_L2=1):
        super().__init__()
        self.n_patches = n_patches
        self.scale_factor = scale_factor
        self.n_patches_L1 = max(2, n_patches // scale_factor)
        self.n_patches_L2 = max(2, self.n_patches_L1 // scale_factor)
        self.drop_rate = drop_rate
        self.num_hops = num_hops
        self.is_directed = is_directed
        self.patch_rw_dim = patch_rw_dim
        self.patch_num_diff = patch_num_diff
        self.metis_enable = metis_enable
        self.num_context = num_context
        self.num_targets = num_targets
        self.num_targets_L1 = num_targets_L1
        self.num_targets_L2 = num_targets_L2

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _diffuse(self, A):
        if self.patch_num_diff == 0:
            return A
        Dinv = A.sum(dim=-1).clamp(min=1).pow(-1).unsqueeze(-1)
        RW = A * Dinv
        M_power = RW
        for _ in range(self.patch_num_diff - 1):
            M_power = torch.matmul(M_power, RW)
        return M_power

    def _coarsen_adj(self, fine_adj, fine_to_coarse, n_coarse):
        """Aggregate a fine-level adjacency matrix to a coarser level.

        Returns coarse_adj[a,b] = sum_{i in a, j in b} fine_adj[i,j].
        """
        n_fine = fine_adj.shape[0]
        M = torch.zeros(n_coarse, n_fine, dtype=fine_adj.dtype)
        M[fine_to_coarse, torch.arange(n_fine)] = 1.0
        return M @ fine_adj @ M.T          # [n_coarse, n_coarse]

    def _metis_partition(self, adj, n_target, n_current):
        """Run METIS on a patch-level adjacency to produce n_target groups.

        Falls back to modulo assignment if METIS is unavailable or fails.
        """
        if not _METIS_AVAILABLE or n_target <= 1 or n_current <= n_target:
            membership = torch.arange(n_current) % max(1, n_target)
        else:
            G = nx.Graph()
            G.add_nodes_from(range(n_current))
            adj_np = (adj > 0).numpy()
            rows, cols = np.where(np.triu(adj_np, k=1))
            if len(rows) > 0:
                G.add_edges_from(zip(rows.tolist(), cols.tolist()))
            try:
                _, membership_list = metis_lib.part_graph(G, n_target, recursive=True)
                membership = torch.tensor(membership_list[:n_current], dtype=torch.long)
            except Exception:
                membership = torch.arange(n_current) % n_target

        max_id = membership.max().item() + 1
        membership = membership + (n_target - max_id)
        return membership          # [n_current], values in [0, n_target)

    # ------------------------------------------------------------------
    # Main callable
    # ------------------------------------------------------------------

    def __call__(self, data):
        data = SubgraphsData(**{k: v for k, v in data})

        # ============================================================
        # Level 0 – fine patches  (identical to GraphJEPAPartitionTransform)
        # ============================================================
        n_L0 = self.n_patches
        if self.metis_enable:
            node_masks, edge_masks = metis_subgraph(
                data, n_patches=n_L0, drop_rate=self.drop_rate,
                num_hops=self.num_hops, is_directed=self.is_directed)
        else:
            node_masks, edge_masks = random_subgraph(
                data, n_patches=n_L0, num_hops=self.num_hops)

        subgraphs_nodes, subgraphs_edges = to_sparse(node_masks, edge_masks)
        combined_subgraphs = combine_subgraphs(
            data.edge_index, subgraphs_nodes, subgraphs_edges,
            num_selected=n_L0, num_nodes=data.num_nodes)

        coarsen_adj_L0 = cal_coarsen_adj(node_masks)   # [n_L0, n_L0]

        if self.patch_rw_dim > 0:
            data.patch_pe = random_walk(coarsen_adj_L0, self.patch_rw_dim)
        if self.patch_num_diff > -1:
            data.coarsen_adj = self._diffuse(coarsen_adj_L0).unsqueeze(0)

        subgraphs_batch = subgraphs_nodes[0]
        mask_L0 = torch.zeros(n_L0).bool()
        mask_L0[subgraphs_batch] = True

        data.subgraphs_batch = subgraphs_batch
        data.subgraphs_nodes_mapper = subgraphs_nodes[1]
        data.subgraphs_edges_mapper = subgraphs_edges[1]
        data.combined_subgraphs = combined_subgraphs
        data.mask = mask_L0.unsqueeze(0)

        # Context / target selection at L0
        unique_L0 = subgraphs_nodes[0].unique().numpy()
        rand_L0 = np.random.choice(unique_L0, self.num_context + self.num_targets, replace=False)
        context_subgraph_idx = rand_L0[0]
        target_subgraph_idxs = torch.tensor(rand_L0[1:])
        data.context_edges_mask = subgraphs_edges[0] == context_subgraph_idx
        data.target_edges_mask = torch.isin(subgraphs_edges[0], target_subgraph_idxs)
        data.context_nodes_mapper = subgraphs_nodes[1, subgraphs_nodes[0] == context_subgraph_idx]
        data.target_nodes_mapper = subgraphs_nodes[1, torch.isin(subgraphs_nodes[0], target_subgraph_idxs)]
        data.context_nodes_subgraph = subgraphs_nodes[0, subgraphs_nodes[0] == context_subgraph_idx]
        data.target_nodes_subgraph = subgraphs_nodes[0, torch.isin(subgraphs_nodes[0], target_subgraph_idxs)]
        data.context_subgraph_idx = context_subgraph_idx.tolist()
        data.target_subgraph_idxs = target_subgraph_idxs.tolist()
        data.call_n_patches = [n_L0]

        # ============================================================
        # Level 1 – medium patches
        # ============================================================
        n_L1 = self.n_patches_L1
        fine_to_medium = self._metis_partition(coarsen_adj_L0, n_L1, n_L0)
        # fine_to_medium: [n_L0], values in [0, n_L1)

        coarsen_adj_L1 = self._coarsen_adj(coarsen_adj_L0, fine_to_medium, n_L1)

        if self.patch_rw_dim > 0:
            data.patch_pe_L1 = random_walk(coarsen_adj_L1, self.patch_rw_dim)
        data.coarsen_adj_L1 = self._diffuse(coarsen_adj_L1).unsqueeze(0)
        data.mask_L1 = torch.ones(n_L1, dtype=torch.bool).unsqueeze(0)
        data.fine_to_medium = fine_to_medium
        data.n_patches_L1 = [n_L1]

        # Context / target selection at L1
        all_L1 = np.arange(n_L1)
        n_sel_L1 = min(self.num_context + self.num_targets_L1, n_L1)
        rand_L1 = np.random.choice(all_L1, n_sel_L1, replace=False)
        data.context_subgraph_idx_L1 = int(rand_L1[0])
        data.target_subgraph_idxs_L1 = rand_L1[1:].tolist()

        # ============================================================
        # Level 2 – coarse patches
        # ============================================================
        n_L2 = self.n_patches_L2
        medium_to_coarse = self._metis_partition(coarsen_adj_L1, n_L2, n_L1)
        # medium_to_coarse: [n_L1], values in [0, n_L2)

        coarsen_adj_L2 = self._coarsen_adj(coarsen_adj_L1, medium_to_coarse, n_L2)

        if self.patch_rw_dim > 0:
            data.patch_pe_L2 = random_walk(coarsen_adj_L2, self.patch_rw_dim)
        data.coarsen_adj_L2 = self._diffuse(coarsen_adj_L2).unsqueeze(0)
        data.mask_L2 = torch.ones(n_L2, dtype=torch.bool).unsqueeze(0)
        data.medium_to_coarse = medium_to_coarse
        data.n_patches_L2 = [n_L2]

        # Context / target selection at L2
        all_L2 = np.arange(n_L2)
        n_sel_L2 = min(self.num_context + self.num_targets_L2, n_L2)
        rand_L2 = np.random.choice(all_L2, n_sel_L2, replace=False)
        data.context_subgraph_idx_L2 = int(rand_L2[0])
        data.target_subgraph_idxs_L2 = rand_L2[1:].tolist()

        data.__num_nodes__ = data.num_nodes
        return data