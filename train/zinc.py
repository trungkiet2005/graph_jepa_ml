import sys
import torch
import numpy as np
import torch.nn.functional as F
from core.config import cfg, update_cfg
from core.get_data import create_dataset
from core.get_model import create_model
from core.trainer import run
from core.model_utils.hyperbolic_dist import hyperbolic_dist
from core.model import GraphHMSJepa


def _compute_loss(model, data, criterion, criterion_type):
    """Compute loss for both single-scale (GraphJepa) and HMS models."""
    if isinstance(model, GraphHMSJepa):
        (tx0, ty0), (tx1, ty1), (tx2, ty2) = model(data)
        if criterion_type == 0:
            l0 = criterion(tx0, ty0)
            l1 = criterion(tx1, ty1)
            l2 = criterion(tx2, ty2)
        elif criterion_type == 1:
            l0 = F.mse_loss(tx0, ty0)
            l1 = F.mse_loss(tx1, ty1)
            l2 = F.mse_loss(tx2, ty2)
        else:
            print('Loss function not supported! Exiting!')
            sys.exit(1)
        w = model.loss_weights
        loss = w[0] * l0 + w[1] * l1 + w[2] * l2
        # VICReg variance regularisation
        loss = loss + model.var_weight * torch.mean(
            torch.relu(1.0 - tx0.detach().std(dim=0)))
        num_t = len(ty0)
    else:
        target_x, target_y = model(data)
        if criterion_type == 0:
            loss = criterion(target_x, target_y)
        elif criterion_type == 1:
            loss = F.mse_loss(target_x, target_y)
        elif criterion_type == 2:
            loss = hyperbolic_dist(target_x, target_y)
        else:
            print('Loss function not supported! Exiting!')
            sys.exit(1)
        num_t = len(target_y)
    return loss, num_t


def _ema_update(model, momentum_weight):
    """Apply EMA update to all target encoders."""
    if isinstance(model, GraphHMSJepa):
        for ctx_enc, tgt_enc in model.encoder_pairs:
            for param_q, param_k in zip(ctx_enc.parameters(), tgt_enc.parameters()):
                param_k.data.mul_(momentum_weight).add_((1. - momentum_weight) * param_q.detach().data)
    else:
        for param_q, param_k in zip(model.context_encoder.parameters(), model.target_encoder.parameters()):
            param_k.data.mul_(momentum_weight).add_((1. - momentum_weight) * param_q.detach().data)


def train(train_loader, model, optimizer, evaluator, device, momentum_weight, sharp=None, criterion_type=0):
    criterion = torch.nn.SmoothL1Loss(beta=0.5)
    step_losses, num_targets = [], []
    for data in train_loader:
        if model.use_lap: # Sign flips for eigenvalue PEs
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


@ torch.no_grad()
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
    cfg.merge_from_file('train/configs/zinc.yaml')
    cfg = update_cfg(cfg)
    run(cfg, create_dataset, create_model, train, test)

