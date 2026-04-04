# Experiment Tracker — Graph-JEPA Improvements

> Goal: Beat the original Graph-JEPA paper and target Oral at A* conference.
> All experiments use **10-fold CV**, reported as **mean +/- std over 5 runs** (matching the paper's protocol).
> Metric: Classification Accuracy (%) for TUD datasets, MAE for ZINC.

---

## Table 1 Comparison — Classification Accuracy (%)

Paper gốc (Table 1, Graph-JEPA, Skenderi et al.):

| Method               | PROTEINS | MUTAG        | DD           | REDDIT-B     | REDDIT-M5    | IMDB-B       | IMDB-M       | ZINC (MAE) |
|----------------------|----------|--------------|--------------|--------------|--------------|--------------|--------------|------------|
| F-GIN (supervised)   | 72.39±2.76 | 90.41±4.61 | 74.87±3.56 | 86.79±2.04 | 53.28±3.17 | 71.83±1.93 | 48.46±2.31 | 0.254±0.005 |
| InfoGraph            | 72.57±0.65 | 87.71±1.77 | 75.23±0.39 | 78.79±2.14 | 51.11±0.55 | 71.11±0.88 | 48.66±0.67 | 0.890±0.017 |
| GraphCL              | 72.86±1.01 | 88.29±1.31 | 74.70±0.70 | 82.63±0.99 | 53.05±0.40 | 70.80±0.77 | 48.49±0.63 | 0.627±0.013 |
| MVGRL                | —        | —            | —            | 84.5±0.6   | —            | 74.2±0.7   | 51.2±0.5   | —          |
| AD-GCL-FIX           | 73.59±0.65 | 89.25±1.45 | 74.49±0.52 | 85.52±0.79 | 53.00±0.82 | 71.57±1.01 | 49.04±0.53 | 0.578±0.012 |
| AD-GCL-OPT           | 73.81±0.46 | 89.70±1.03 | 75.10±0.39 | 85.52±0.79 | 54.93±0.43 | 72.33±0.56 | 49.89±0.66 | 0.544±0.004 |
| GraphMAE             | 75.30±0.39 | 88.19±1.26 | 74.27±1.07 | 88.01±0.19 | 46.06±3.44 | 75.52±0.66 | 51.63±0.52 | 0.935±0.034 |
| S2GAE                | 76.37±0.43 | 88.26±0.76 | —            | 87.83±0.27 | —            | 75.76±0.62 | 51.79±0.36 | —          |
| LaGraph              | 75.2±0.4 | 90.2±1.1   | 78.1±0.4   | 90.4±0.8   | 56.4±0.4   | 73.7±0.9   | —            | —          |
| **Graph-JEPA (paper)** | **75.68±3.78** | **91.25±5.75** | **78.64±2.35** | **91.99±1.59** | **56.73±1.96** | **73.68±3.24** | **50.69±2.91** | **0.434±0.014** |

---

## Our Experiments — MUTAG

| Exp  | Method                            | MUTAG Acc (%)    | Status    | vs Paper    | Notes |
|------|-----------------------------------|------------------|-----------|-------------|-------|
| 01   | HMS-JEPA Baseline (reproduce)     | —                | NOT RUN   | —           | Baseline reproduction |
| 02   | VICReg Regularization (C-JEPA)    | 87.09±6.51*      | 4/5 runs  | -4.16pp     | Loss scale ~4M — VICReg cov term quá mạnh |
| 03   | Layer-wise Attention Pool          | —                | NOT RUN   | —           | — |
| 04   | Multi-view Re-masking             | —                | NOT RUN   | —           | — |
| 05   | Adaptive Loss Weights             | —                | NOT RUN   | —           | — |
| 06   | Combined (VICReg+LayerAttn+Adapt) | —                | NOT RUN   | —           | — |

*exp02: only 4/5 runs completed. Run 4 chưa xong. Avg of runs 0-3: (88.27+86.67+85.15+88.25)/4 = 87.09%

---

## Our Experiments — All Datasets (to be filled as experiments run)

| Exp  | PROTEINS | MUTAG    | DD       | REDDIT-B | REDDIT-M5 | IMDB-B   | IMDB-M   | ZINC     |
|------|----------|----------|----------|----------|-----------|----------|----------|----------|
| 01   | —        | —        | —        | —        | —         | —        | —        | —        |
| 02   | —        | 87.09*   | —        | —        | —         | —        | —        | —        |
| 03   | —        | —        | —        | —        | —         | —        | —        | —        |
| 04   | —        | —        | —        | —        | —         | —        | —        | —        |
| 05   | —        | —        | —        | —        | —         | —        | —        | —        |
| 06   | —        | —        | —        | —        | —         | —        | —        | —        |

---

## Detailed Run Logs

### EXP 02 — VICReg Regularization (C-JEPA) on MUTAG

**Config:** HMS-JEPA, Hadamard, GINEConv, hidden=512, nlayer_gnn=2, nlayer_mixer=4, n_patches=32, epochs=50, lr=0.0005, batch=128, 10-fold CV x 5 runs

**Per-Run Results:**

| Run | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Mean      | Std      |
|-----|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|-----------|----------|
| 0   | 84.21  | 84.21  | 84.21  | 94.74  | 89.47  | 89.47  | 94.74  | 89.47  | 94.44  | 77.78  | **88.27** | 5.34     |
| 1   | 89.47  | 84.21  | 78.95  | 89.47  | 84.21  | 94.74  | 94.74  | 84.21  | 94.44  | 72.22  | **86.67** | 7.02     |
| 2   | 94.74  | 84.21  | 73.68  | 84.21  | 84.21  | 94.74  | 78.95  | 78.95  | 100.00 | 77.78  | **85.15** | 8.18     |
| 3   | —      | —      | —      | —      | —      | —      | —      | —      | —      | —      | **88.25** | 7.64     |
| 4   | —      | —      | —      | —      | —      | —      | —      | —      | —      | —      | (running) | —        |
| **Avg** | | | | | | | | | | | **87.09** | **~7.05** |

**Observations:**
- Loss scale ~4,000,000 — rất lớn, VICReg covariance term đang dominate
- Accuracy 87.09% < Paper's 91.25% — **cần giảm cov_weight hoặc vicreg scale**
- High variance across folds (5-8%) — consistent with paper (5.75%)

**Action items:**
- [ ] Giảm `cov_weight` từ 0.05 xuống 0.01 hoặc 0.005
- [ ] Giảm VICReg scale từ 0.01 xuống 0.001
- [ ] Thử bỏ VICReg trên target (chỉ regularize prediction)

---

## Ablation Reference (Paper Table 3 — Distance Function)

| Dataset | Ours (Hyperbolic) | Euclidean | Hyperbolic (Poincaré) | Euclidean (LD) | Hyperbolic (LD) |
|---------|-------------------|-----------|----------------------|----------------|-----------------|
| MUTAG   | 91.25±5.75        | 87.04±6.01 | 89.43±5.67          | 86.63±5.9      | 86.32±5.52      |
| REDDIT-M| 56.73±1.96        | 56.55±1.94 | 56.19±1.95          | 54.84±1.6      | 55.07±1.83      |
| IMDB-B  | 73.68±3.24        | 73.76±3.46 | NaN                  | 72.5±3.97      | 73.4±4.07       |
| ZINC    | 0.434±0.01        | 0.471±0.01 | 0.605±0.01           | 0.952±0.05     | 0.912±0.04      |

---

## Dataset Statistics (Paper Table 7)

| Dataset    | Graphs | Classes | Avg Nodes | Avg Edges |
|------------|--------|---------|-----------|-----------|
| PROTEINS   | 1,113  | 2       | 39.06     | 72.82     |
| MUTAG      | 188    | 2       | 17.93     | 19.79     |
| DD         | 1,178  | 2       | 284.32    | 715.66    |
| REDDIT-B   | 2,000  | 2       | 429.63    | 497.75    |
| REDDIT-M5  | 4,999  | 5       | 508.52    | 594.87    |
| IMDB-B     | 1,000  | 2       | 19.77     | 96.53     |
| IMDB-M     | 1,500  | 3       | 13.00     | 65.94     |
| ZINC       | 12,000 | 0 (reg) | 23.2      | 49.8      |

---

## Hyperparameters Reference (Paper Table 8)

| Hyperparameter     | PROTEINS | MUTAG | DD | REDDIT-B | REDDIT-M5 | IMDB-B | IMDB-M | ZINC |
|-------------------|----------|-------|----|----------|-----------|--------|--------|------|
| Num Subgraphs     | 32       | 32    | 32 | 128      | 128       | 32     | 32     | 32   |
| GNN Layers        | 2        | 2     | 3  | 2        | 2         | 2      | 2      | 2    |
| Encoder Blocks    | 4        | 4     | 4  | 4        | 4         | 4      | 4      | 4    |
| Embedding Size    | 512      | 512   | 512| 512      | 512       | 512    | 512    | 512  |
| RWSE Size         | 20       | 15    | 30 | 40       | 40        | 15     | 15     | 20   |
| Context - Targets | 1-2      | 1-3   | 1-4| 1-4      | 1-4       | 1-4    | 1-4    | 1-4  |
