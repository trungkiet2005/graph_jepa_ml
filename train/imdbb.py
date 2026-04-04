import torch
import numpy as np
import torch.nn.functional as F
from core.config import cfg, update_cfg
from core.get_data import create_dataset
from core.get_model import create_model
from core.trainer import run_k_fold
from train.zinc import _compute_loss, _ema_update


def train(train_loader, model, optimizer, evaluator, device, momentum_weight, sharp=None, criterion_type=0):
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
    epoch_loss = np.average(step_losses, weights=num_targets)
    return None, epoch_loss


@torch.no_grad()
def test(loader, model, evaluator, device, criterion_type=0):
    criterion = torch.nn.SmoothL1Loss(beta=0.5)
    step_losses, num_targets = [], []
    for data in loader:
        data = data.to(device)
        loss, num_t = _compute_loss(model, data, criterion, criterion_type)
        step_losses.append(loss.item())
        num_targets.append(num_t)
    epoch_loss = np.average(step_losses, weights=num_targets)
    return None, epoch_loss


if __name__ == '__main__':
    cfg.merge_from_file('train/configs/imdbb.yaml')
    cfg = update_cfg(cfg)
    cfg.k = 10
    run_k_fold(cfg, create_dataset, create_model, train, test)
