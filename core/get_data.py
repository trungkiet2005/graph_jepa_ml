import sys

import numpy as np
import torch
from torch_geometric.transforms import Compose, Constant
from core.config import cfg, update_cfg

from torch_geometric.datasets import ZINC, TUDataset
from core.data_utils.exp import PlanarSATPairsDataset
from core.transform import PositionalEncodingTransform, GraphJEPAPartitionTransform, GraphHMSJEPAPartitionTransform

# Short names used in some experiment scripts → TUDataset `name=` (PyG)
_TUD_DATASET_ALIASES = {
    'IMDB-B': 'IMDB-BINARY',
    'IMDB-M': 'IMDB-MULTI',
}


def calculate_stats(dataset):
    num_graphs = len(dataset)
    ave_num_nodes = np.array([g.num_nodes for g in dataset]).mean()
    ave_num_edges = np.array([g.num_edges for g in dataset]).mean()
    print(
        f'# Graphs: {num_graphs}, average # nodes per graph: {ave_num_nodes}, average # edges per graph: {ave_num_edges}.')


def create_dataset(cfg):
    ds_name = str(cfg.dataset)
    if ds_name in _TUD_DATASET_ALIASES:
        cfg.defrost()
        cfg.dataset = _TUD_DATASET_ALIASES[ds_name]
        cfg.freeze()

    pre_transform = PositionalEncodingTransform(
        rw_dim=cfg.pos_enc.rw_dim, lap_dim=cfg.pos_enc.lap_dim)

    transform_train = transform_eval = None

    if cfg.metis.n_patches > 0:
        use_hms = getattr(cfg.jepa, 'num_scales', 1) > 1
        _TransformCls = GraphHMSJEPAPartitionTransform if use_hms else GraphJEPAPartitionTransform

        common_kwargs = dict(
            n_patches=cfg.metis.n_patches,
            patch_rw_dim=cfg.pos_enc.patch_rw_dim,
            patch_num_diff=cfg.pos_enc.patch_num_diff,
            num_context=cfg.jepa.num_context,
            num_targets=cfg.jepa.num_targets,
        )
        if use_hms:
            common_kwargs.update(dict(
                metis_enable=cfg.metis.enable,
                scale_factor=cfg.jepa.scale_factor,
                num_hops=cfg.metis.num_hops,
                is_directed=cfg.dataset == 'TreeDataset',
                num_targets_L1=cfg.jepa.num_targets_L1,
                num_targets_L2=cfg.jepa.num_targets_L2,
            ))
        else:
            common_kwargs.update(dict(
                metis=cfg.metis.enable,
                num_hops=cfg.metis.num_hops,
                is_directed=cfg.dataset == 'TreeDataset',
            ))

        _transform_train = _TransformCls(drop_rate=cfg.metis.drop_rate, **common_kwargs)
        _transform_eval  = _TransformCls(drop_rate=0.0, **common_kwargs)
        transform_train  = _transform_train
        transform_eval   = _transform_eval
    else:
        print('Not supported...')
        sys.exit(1)

    if cfg.dataset == 'ZINC':
        root = 'dataset/ZINC'
        train_dataset = ZINC(
            root, subset=True, split='train', pre_transform=pre_transform, transform=transform_train)
        val_dataset = ZINC(root, subset=True, split='val',
                           pre_transform=pre_transform, transform=transform_eval)
        test_dataset = ZINC(root, subset=True, split='test',
                            pre_transform=pre_transform, transform=transform_eval)
    elif cfg.dataset in ['PROTEINS', 'MUTAG', 'DD', 'REDDIT-BINARY', 'REDDIT-MULTI-5K', 'IMDB-BINARY', 'IMDB-MULTI']:
        if cfg.dataset not in ['PROTEINS', 'MUTAG', 'DD']:
            pre_transform = Compose([Constant(value=0, cat=False), pre_transform])

        dataset = TUDataset(root='dataset/TUD', name=cfg.dataset, pre_transform=pre_transform)

        return dataset, transform_train, transform_eval

    elif cfg.dataset == 'exp-classify':
        root = "dataset/EXP/"
        dataset = PlanarSATPairsDataset(root, pre_transform=pre_transform)
        return dataset, transform_train, transform_eval

    else:
        print(f"Dataset not supported: {cfg.dataset!r}")
        sys.exit(1)

    torch.set_num_threads(cfg.num_workers)
    if not cfg.metis.online:
        train_dataset = [x for x in train_dataset]
    val_dataset = [x for x in val_dataset]
    test_dataset = [x for x in test_dataset]

    return train_dataset, val_dataset, test_dataset


if __name__ == '__main__':
    print("Generating data")

    cfg.merge_from_file('train/configs/zinc.yaml')
    cfg = update_cfg(cfg)
    cfg.metis.n_patches = 0
    train_dataset, val_dataset, test_dataset = create_dataset(cfg)

    if cfg.dataset == 'exp-classify':
        print('------------Dataset--------------')
        calculate_stats(train_dataset)
        print('------------------------------')
    else:
        print('------------Train--------------')
        calculate_stats(train_dataset)
        print('------------Validation--------------')
        calculate_stats(val_dataset)
        print('------------Test--------------')
        calculate_stats(test_dataset)
        print('------------------------------')
