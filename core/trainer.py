import os
import torch
import random
import time
import numpy as np
from core.log import config_logger
from core.asam import ASAM
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run(cfg, create_dataset, create_model, train, test, evaluator=None):
    if cfg.seed is not None:
        seeds = [cfg.seed]
        cfg.train.runs = 1
    else:
        seeds = [21, 42, 41, 95, 12, 35, 66, 85, 3, 1234]

    writer, logger = config_logger(cfg)

    train_dataset, val_dataset, test_dataset = create_dataset(cfg)

    _pw = cfg.num_workers > 0
    train_loader = DataLoader(
        train_dataset, cfg.train.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, persistent_workers=_pw)
    val_loader = DataLoader(
        val_dataset,  cfg.train.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True, persistent_workers=_pw)
    test_loader = DataLoader(
        test_dataset, cfg.train.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True, persistent_workers=_pw)

    hms_tag = "HMS-JEPA" if getattr(cfg.jepa, 'num_scales', 1) > 1 else "Graph-JEPA"
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT: {cfg.dataset} (regression) | {cfg.train.runs} runs")
    print(f"  Model: {cfg.model.gnn_type} + {cfg.model.gMHA_type} | hidden={cfg.model.hidden_size}")
    print(f"  Architecture: {hms_tag} | n_patches={cfg.metis.n_patches}")
    print(f"{'='*70}")

    train_losses = []
    per_epoch_times = []
    total_times = []
    maes = []
    for run in range(cfg.train.runs):
        set_seed(seeds[run])
        model = create_model(cfg).to(cfg.device)
        n_params = count_parameters(model)

        print(f"\n{'─'*70}")
        print(f"  RUN {run+1}/{cfg.train.runs}  (seed={seeds[run]}, params={n_params:,})")
        print(f"{'─'*70}")

        if cfg.train.optimizer == 'ASAM':
            sharp = True
            optimizer = torch.optim.SGD(
                model.parameters(), lr=cfg.train.lr, momentum=0.9, weight_decay=cfg.train.wd)
            minimizer = ASAM(optimizer, model, rho=0.5)
        else:
            sharp = False
            optimizer = torch.optim.Adam(
                model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.wd)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                               factor=cfg.train.lr_decay,
                                                               patience=cfg.train.lr_patience,
                                                               )

        start_outer = time.time()
        per_epoch_time = []

        # Create EMA scheduler for target encoder param update
        ipe = len(train_loader)
        ema_params = [0.996, 1.0]
        momentum_scheduler = (ema_params[0] + i*(ema_params[1]-ema_params[0])/(ipe*cfg.train.epochs)
                            for i in range(int(ipe*cfg.train.epochs)+1))
        for epoch in range(cfg.train.epochs):
            start = time.time()
            model.train()
            _, train_loss = train(
                train_loader, model, optimizer if not sharp else minimizer,
                    evaluator=evaluator, device=cfg.device, momentum_weight=next(momentum_scheduler),
                    sharp=sharp, criterion_type=cfg.jepa.dist)
            model.eval()
            _, val_loss = test(val_loader, model,
                                      evaluator=evaluator, device=cfg.device, criterion_type=cfg.jepa.dist)
            _, test_loss = test(test_loader, model,
                                      evaluator=evaluator, device=cfg.device, criterion_type=cfg.jepa.dist)

            time_cur_epoch = time.time() - start
            per_epoch_time.append(time_cur_epoch)

            # Compact logging
            if epoch == 0 or epoch == cfg.train.epochs - 1 or (epoch + 1) % 10 == 0:
                lr_now = optimizer.param_groups[0]['lr']
                print(f'  [R{run+1}] E{epoch:03d}  train={train_loss:.4f}  '
                      f'val={val_loss:.4f}  test={test_loss:.4f}  '
                      f'lr={lr_now:.6f}  {time_cur_epoch:.1f}s')

            writer.add_scalar(f'Run{run}/train-loss', train_loss, epoch)
            writer.add_scalar(f'Run{run}/val-loss', val_loss, epoch)

            if scheduler is not None:
                scheduler.step(val_loss)

            if not sharp:
                if optimizer.param_groups[0]['lr'] < cfg.train.min_lr:
                    print(f'  [R{run+1}] Early stop at epoch {epoch} (LR < min_lr)')
                    break

        per_epoch_time = np.mean(per_epoch_time)
        total_time = (time.time()-start_outer)/3600

        model.eval()
        X_train, y_train = [], []
        X_test, y_test = [], []
        for data in train_loader:
            data.to(cfg.device)
            with torch.no_grad():
                features = model.encode(data)
                X_train.append(features.detach().cpu().numpy())
                y_train.append(data.y.detach().cpu().numpy())

        X_train = np.concatenate(X_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)

        for data in test_loader:
            data.to(cfg.device)
            with torch.no_grad():
                features = model.encode(data)
                X_test.append(features.detach().cpu().numpy())
                y_test.append(data.y.detach().cpu().numpy())

        X_test = np.concatenate(X_test, axis=0)
        y_test = np.concatenate(y_test, axis=0)

        # Fine tuning on the learned representations via Ridge Regression
        lin_model = Ridge()
        lin_model.fit(X_train, y_train)
        lin_predictions = lin_model.predict(X_test)
        lin_mae = mean_absolute_error(y_test, lin_predictions)
        maes.append(lin_mae)

        r2_train = lin_model.score(X_train, y_train)
        print(f"\n  ┌─ Run {run+1} Summary ─────────────────────────────────────────┐")
        print(f"  │ MAE: {lin_mae:.4f}  |  Train R2: {r2_train:.4f}")
        print(f"  │ Loss: {train_loss:.4f} (train)  |  Epochs: {epoch+1}")
        print(f"  │ Time: {per_epoch_time:.2f}s/epoch, {total_time:.4f}h total")
        print(f"  │ Features: dim={X_train.shape[1]}, train_n={len(y_train)}, test_n={len(y_test)}")
        print(f"  └────────────────────────────────────────────────────────┘")

        train_losses.append(train_loss)
        per_epoch_times.append(per_epoch_time)
        total_times.append(total_time)

    # Final summary
    maes = np.array(maes)
    print(f"\n{'='*70}")
    print(f"  FINAL RESULTS — {cfg.dataset} (regression)")
    print(f"{'='*70}")
    print(f"  Per-Run MAE: {', '.join([f'{m:.4f}' for m in maes])}")
    print(f"\n  ┌─────────────────────────────────────────────────┐")
    print(f"  │  {cfg.dataset} MAE: {maes.mean():.4f} +/- {maes.std():.4f}         │")
    print(f"  └─────────────────────────────────────────────────┘")
    print(f"\n  [TRACKER] {cfg.dataset}: MAE={maes.mean():.4f}+/-{maes.std():.4f}  "
          f"({cfg.train.runs} runs, {cfg.train.epochs}ep)")

    if cfg.train.runs > 1:
        train_loss_t = torch.tensor(train_losses)
        per_epoch_time_t = torch.tensor(per_epoch_times)
        total_time_t = torch.tensor(total_times)
        print(f"  Train Loss: {train_loss_t.mean():.4f} +/- {train_loss_t.std():.4f}")
        print(f"  Seconds/epoch: {per_epoch_time_t.mean():.4f}")
        print(f"  Hours/total: {total_time_t.mean():.4f}")
        logger.info("-"*50)
        logger.info(cfg)
        logger.info(f'\nFinal Train Loss: {train_loss_t.mean():.4f} +/- {train_loss_t.std():.4f}'
                    f'\nSeconds/epoch: {per_epoch_time_t.mean():.4f}'
                    f'\nHours/total: {total_time_t.mean():.4f}')
        logger.info(f'{cfg.dataset} MAE: {maes.mean():.4f} +/- {maes.std():.4f}')

    print(f"{'='*70}\n")

def count_parameters(model):
    # For counting number of parameteres: need to remove unnecessary DiscreteEncoder, and other additional unused params
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _graph_labels_numpy(dataset):
    """Graph-level labels for stratified split (avoids InMemoryDataset.data access warning)."""
    y = getattr(dataset, "y", None)
    if y is None:
        y = dataset.data.y
    if isinstance(y, torch.Tensor):
        y = y.view(-1).detach().cpu().numpy()
    else:
        y = np.asarray(y).reshape(-1)
    return y


def k_fold(dataset, folds=10):
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)
    train_indices, test_indices = [], []
    ys = _graph_labels_numpy(dataset)
    for train, test in skf.split(torch.zeros(len(dataset)), ys):
        train_indices.append(torch.from_numpy(train).to(torch.long))
        test_indices.append(torch.from_numpy(test).to(torch.long))
    return train_indices, test_indices


def run_k_fold(cfg, create_dataset, create_model, train, test, evaluator=None, k=10):
    if cfg.seed is not None:
        seeds = [cfg.seed]
        cfg.train.runs = 1
    else:
        seeds = [42, 21, 95, 12, 35]

    writer, logger = config_logger(cfg)
    dataset, transform, transform_eval = create_dataset(cfg)

    if hasattr(dataset, 'train_indices'):
        k_fold_indices = dataset.train_indices, dataset.test_indices
    else:
        k_fold_indices = k_fold(dataset, cfg.k)

    n_folds = len(k_fold_indices[0])
    n_params = None

    print(f"\n{'='*70}")
    print(f"  EXPERIMENT: {cfg.dataset} | {cfg.train.runs} runs x {n_folds}-fold CV")
    print(f"  Model: {cfg.model.gnn_type} + {cfg.model.gMHA_type} | hidden={cfg.model.hidden_size}")
    print(f"  Epochs: {cfg.train.epochs} | LR: {cfg.train.lr} | Batch: {cfg.train.batch_size}")
    hms_tag = "HMS-JEPA" if getattr(cfg.jepa, 'num_scales', 1) > 1 else "Graph-JEPA"
    print(f"  Architecture: {hms_tag} | n_patches={cfg.metis.n_patches}")
    print(f"{'='*70}")

    train_losses = []
    per_epoch_times = []
    total_times = []
    run_metrics = []
    all_fold_accs = []   # store per-fold accuracy for all runs

    for run in range(cfg.train.runs):
        set_seed(seeds[run])
        acc = []

        print(f"\n{'─'*70}")
        print(f"  RUN {run+1}/{cfg.train.runs}  (seed={seeds[run]})")
        print(f"{'─'*70}")

        for fold, (train_idx, test_idx) in enumerate(zip(*k_fold_indices)):
            train_dataset = dataset[train_idx]
            test_dataset = dataset[test_idx]
            train_dataset.transform = transform
            test_dataset.transform = transform_eval
            test_dataset = [x for x in test_dataset]

            if not cfg.metis.online:
                train_dataset = [x for x in train_dataset]

            _pw = cfg.num_workers > 0
            train_loader = DataLoader(
                train_dataset, cfg.train.batch_size, shuffle=True,
                num_workers=cfg.num_workers, pin_memory=True, persistent_workers=_pw)
            test_loader = DataLoader(
                test_dataset,  cfg.train.batch_size, shuffle=False,
                num_workers=cfg.num_workers, pin_memory=True, persistent_workers=_pw)

            model = create_model(cfg).to(cfg.device)
            if n_params is None:
                n_params = count_parameters(model)
                print(f"  Parameters: {n_params:,}")

            optimizer = torch.optim.Adam(
                model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.wd)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                   factor=cfg.train.lr_decay,
                                                                   patience=cfg.train.lr_patience,
                                                                   )

            start_outer = time.time()
            per_epoch_time = []

            # Create EMA scheduler for target encoder param update
            ipe = len(train_loader)
            ema_params = [0.996, 1.0]
            momentum_scheduler = (ema_params[0] + i*(ema_params[1]-ema_params[0])/(ipe*cfg.train.epochs)
                                for i in range(int(ipe*cfg.train.epochs)+1))

            final_train_loss = 0
            final_test_loss = 0
            final_epoch = 0

            for epoch in range(cfg.train.epochs):
                start = time.time()
                model.train()
                _, train_loss = train(
                    train_loader, model, optimizer,
                    evaluator=evaluator, device=cfg.device,
                    momentum_weight=next(momentum_scheduler), criterion_type=cfg.jepa.dist)
                model.eval()
                _, test_loss = test(
                    test_loader, model, evaluator=evaluator, device=cfg.device,
                    criterion_type=cfg.jepa.dist)

                scheduler.step(test_loss)
                time_cur_epoch = time.time() - start
                per_epoch_time.append(time_cur_epoch)

                final_train_loss = train_loss
                final_test_loss = test_loss
                final_epoch = epoch

                # Compact logging: only first, last, every 10th epoch, or LR drop
                if epoch == 0 or epoch == cfg.train.epochs - 1 or (epoch + 1) % 10 == 0:
                    lr_now = optimizer.param_groups[0]['lr']
                    print(f'  [R{run+1}F{fold}] E{epoch:03d}  '
                          f'train={train_loss:.4f}  test={test_loss:.4f}  '
                          f'lr={lr_now:.6f}  {time_cur_epoch:.1f}s')

                writer.add_scalar(f'Run{run}/train-loss', train_loss, epoch)
                writer.add_scalar(f'Run{run}/test-loss', test_loss, epoch)

                if optimizer.param_groups[0]['lr'] < cfg.train.min_lr:
                    print(f'  [R{run+1}F{fold}] Early stop at epoch {epoch} (LR < min_lr)')
                    break

            per_epoch_time = np.mean(per_epoch_time)
            total_time = (time.time()-start_outer)/3600

            # Extract features and evaluate with linear classifier
            model.eval()
            X_train, y_train = [], []
            X_test, y_test = [], []

            for data in train_loader:
                data.to(cfg.device)
                with torch.no_grad():
                    features = model.encode(data)
                    X_train.append(features.detach().cpu().numpy())
                    y_train.append(data.y.detach().cpu().numpy())

            X_train = np.concatenate(X_train, axis=0)
            y_train = np.concatenate(y_train, axis=0)

            for data in test_loader:
                data.to(cfg.device)
                with torch.no_grad():
                    features = model.encode(data)
                    X_test.append(features.detach().cpu().numpy())
                    y_test.append(data.y.detach().cpu().numpy())

            X_test = np.concatenate(X_test, axis=0)
            y_test = np.concatenate(y_test, axis=0)

            lin_model = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=5000, solver="lbfgs", random_state=0),
            )
            lin_model.fit(X_train, y_train)
            lin_predictions = lin_model.predict(X_test)
            lin_accuracy = accuracy_score(y_test, lin_predictions)
            acc.append(lin_accuracy)

            print(f'  [R{run+1}F{fold}] >> Acc: {lin_accuracy*100:.2f}%  '
                  f'(train_n={len(y_train)}, test_n={len(y_test)}, feat_dim={X_train.shape[1]})  '
                  f'{per_epoch_time:.2f}s/ep  {final_epoch+1}ep')

            train_losses.append(train_loss)
            per_epoch_times.append(per_epoch_time)
            total_times.append(total_time)

        acc = np.array(acc)
        run_metrics.append([acc.mean(), acc.std()])
        all_fold_accs.append(acc)

        # Per-run summary
        print(f"\n  ┌─ Run {run+1} Summary ─────────────────────────────────────────┐")
        fold_strs = '  '.join([f'F{i}={a*100:.1f}' for i, a in enumerate(acc)])
        print(f"  │ Folds: {fold_strs}")
        print(f"  │ Mean Acc: {acc.mean()*100:.2f}% +/- {acc.std()*100:.2f}%")
        print(f"  │ Loss: {final_train_loss:.4f} (train) / {final_test_loss:.4f} (test)")
        print(f"  │ Time: {per_epoch_time:.2f}s/epoch, {total_time:.4f}h total")
        print(f"  └────────────────────────────────────────────────────────┘")

    # ════════════════════════════════════════════════════════════════
    # FINAL SUMMARY — designed for easy copy into tracker.md
    # ════════════════════════════════════════════════════════════════
    run_metrics = np.array(run_metrics)
    all_fold_accs = np.array(all_fold_accs)  # [n_runs, n_folds]

    print(f"\n{'='*70}")
    print(f"  FINAL RESULTS — {cfg.dataset}")
    print(f"{'='*70}")
    print(f"  Architecture : {hms_tag}")
    print(f"  Parameters   : {n_params:,}")
    print(f"  Protocol     : {cfg.train.runs} runs x {n_folds}-fold CV")
    print(f"  Epochs       : {cfg.train.epochs}")

    # Per-run table
    print(f"\n  Per-Run Accuracy (%):")
    print(f"  {'Run':<6}", end='')
    for f in range(n_folds):
        print(f"{'F'+str(f):>7}", end='')
    print(f"  {'Mean':>7}  {'Std':>6}")
    print(f"  {'─'*6}", end='')
    for _ in range(n_folds):
        print(f"{'─'*7}", end='')
    print(f"  {'─'*7}  {'─'*6}")

    for r in range(len(all_fold_accs)):
        print(f"  {r+1:<6}", end='')
        for f in range(n_folds):
            print(f"{all_fold_accs[r,f]*100:>7.2f}", end='')
        print(f"  {run_metrics[r,0]*100:>7.2f}  {run_metrics[r,1]*100:>6.2f}")

    # Overall
    overall_mean = run_metrics[:, 0].mean() * 100
    overall_std = run_metrics[:, 0].std() * 100
    avg_inner_std = run_metrics[:, 1].mean() * 100

    print(f"\n  ┌─────────────────────────────────────────────────┐")
    print(f"  │  {cfg.dataset} ACCURACY: {overall_mean:.2f} +/- {overall_std:.2f}%  │")
    print(f"  │  (avg inner std: {avg_inner_std:.2f}%)                      │")
    print(f"  └─────────────────────────────────────────────────┘")

    # Markdown-ready line for tracker
    print(f"\n  [TRACKER] {cfg.dataset}: {overall_mean:.2f}+/-{overall_std:.2f}%  "
          f"({cfg.train.runs} runs, {n_folds}-fold CV, {cfg.train.epochs}ep)")

    if cfg.train.runs > 1:
        train_loss_t = torch.tensor(train_losses)
        per_epoch_time_t = torch.tensor(per_epoch_times)
        total_time_t = torch.tensor(total_times)
        print(f"  Train Loss: {train_loss_t.mean():.4f} +/- {train_loss_t.std():.4f}")
        print(f"  Seconds/epoch: {per_epoch_time_t.mean():.4f}")
        print(f"  Hours/total: {total_time_t.mean():.4f}")
        logger.info("-"*50)
        logger.info(cfg)
        logger.info(f'\nFinal Train Loss: {train_loss_t.mean():.4f} +/- {train_loss_t.std():.4f}'
                    f'\nSeconds/epoch: {per_epoch_time_t.mean():.4f}'
                    f'\nHours/total: {total_time_t.mean():.4f}')
        logger.info(f'{cfg.dataset} ACCURACY: {overall_mean:.2f} +/- {overall_std:.2f}%')

    print(f"{'='*70}\n")
