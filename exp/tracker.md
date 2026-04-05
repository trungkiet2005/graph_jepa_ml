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
| 05   | Adaptive Loss Weights (Kendall)   | 86.67±1.33       | DONE (4 runs) | −4.58pp | Kaggle 2026-04-04; `train.runs: 5` nhưng log chỉ có 4 run hoàn chỉnh |
| 06   | Combined (VICReg+LayerAttn+Adapt) | 84.30±1.72       | DONE (4 runs) | −6.95pp | Kaggle 2026-04-04; `train.runs: 5` nhưng log chỉ có 4 run hoàn chỉnh |
| 07   | H100 All-datasets baseline         | 85.68±1.50       | PARTIAL   | −5.57pp     | Kaggle H100; MUTAG hoàn tất, job dừng ở PROTEINS do `NameError: exit` |

*exp02: only 4/5 runs completed. Run 4 chưa xong. Avg of runs 0-3: (88.27+86.67+85.15+88.25)/4 = 87.09%

---

## Our Experiments — All Datasets (to be filled as experiments run)

| Exp  | PROTEINS | MUTAG    | DD       | REDDIT-B | REDDIT-M5 | IMDB-B   | IMDB-M   | ZINC     |
|------|----------|----------|----------|----------|-----------|----------|----------|----------|
| 01   | —        | —        | —        | —        | —         | —        | —        | —        |
| 02   | —        | 87.09*   | —        | —        | —         | —        | —        | —        |
| 03   | —        | —        | —        | —        | —         | —        | —        | —        |
| 04   | —        | —        | —        | —        | —         | —        | —        | —        |
| 05   | —        | 86.67±1.33 | —        | —        | —         | —        | —        | —        |
| 06   | —        | 84.30±1.72 | —        | —        | —         | —        | —        | —        |
| 07   | CRASH (`NameError: exit`) | 85.68±1.50 | —        | —        | —         | —        | —        | —        |

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

### EXP 05 — Adaptive Loss Weights (Kendall uncertainty) on MUTAG

**Setup:** HMS-JEPA với trọng số loss học theo homoscedastic uncertainty (Kendall et al., CVPR 2018): `L = 0.5 Σ exp(−log σ²_i) L_i + 0.5 Σ log σ²_i`; `GraphHMSJepaAdaptive` + `SmoothL1` JEPA; giữ variance regularization trên target L0 như baseline (`var_weight: 0.01`). **Kaggle**, 2026-04-04.

**Config:** `dataset: MUTAG`, Hadamard, GINEConv, hidden=512, nlayer_gnn=2, nlayer_mlpmixer=4, n_patches=32, epochs=50, lr=5e-4, batch=128, k=10, `jepa.loss_weights: [1,0.5,0.25]` (trong cfg; objective train dùng Kendall adaptive), `train.runs: 5` (log chỉ **4 run** có dòng `Acc mean` đầy đủ).

**Aggregate (mean ± std của 4 giá trị mean accuracy theo run):** **86.67±1.33%** (paper Graph-JEPA: **91.25±5.75%**).

**Per-run summary (`Acc mean` / `std` trên 10 fold):**

| Run | Mean acc | Std (across folds) |
|-----|----------|--------------------|
| 0   | 86.70%   | 9.70%              |
| 1   | 86.70%   | 6.46%              |
| 2   | 88.27%   | 8.36%              |
| 3   | 85.03%   | 9.06%              |
| **Mean of run means** | **86.67%** | **1.33%** (std of 4 run means) |

**Per-fold accuracies — Run 0 only** (%): 73.68, 84.21, 73.68, 89.47, 94.74, 94.74, 89.47, 94.74, 100.00, 72.22.

**Observations:**
- Adaptive-only **cao hơn** EXP 06 combined (~86.7% vs ~84.3%) và **gần** EXP 02 VICReg (~87.1%); vẫn **thấp hơn** paper ~4.6 điểm phần trăm (mean).
- Precision học được ~**L0≈1.99, L1≈1.00, L2≈0.50** (≈ tỉ lệ 2:1:0.5), sát prior cố định `[1, 0.5, 0.25]` — uncertainty weighting **không** thay đổi mạnh cân bằng so với baseline.
- Std giữa các fold trong một run vẫn lớn (≈6–10%); nên hoàn thành run thứ 5 hoặc cố định seed nếu so sánh với paper.

---

### EXP 06 — Combined: VICReg + LayerAttn + Adaptive Weights (MUTAG)

**Setup:** HMS-JEPA + layer-wise attention pooling + Kendall uncertainty weights + VICReg trên `pred_L0/1/2` (var+cov), SmoothL1 JEPA loss. **Kaggle**, 2026-04-04.

**Config:** `dataset: MUTAG`, Hadamard, GINEConv, hidden=512, nlayer_gnn=2, nlayer_mlpmixer=4, n_patches=32, epochs=50, lr=5e-4, batch=128, k=10, `train.runs: 5` (chỉ **4 run** có summary đầy đủ trong log).

**Aggregate (mean ± std of the 4 per-run accuracies):** **84.30±1.72%** (paper Graph-JEPA: **91.25±5.75%**).

**Per-run summary (library `Acc mean` / `std` over 10 folds):**

| Run | Mean acc | Std (across folds) |
|-----|----------|--------------------|
| 0   | 86.17%   | 7.23%              |
| 1   | 82.43%   | 6.83%              |
| 2   | 83.98%   | 5.51%              |
| 3   | 85.61%   | 9.47%              |
| **Mean of run means** | **84.30%** | **1.72%** (std of 4 run means) |

**Per-fold accuracies — Run 0 only** (đủ 10 fold; các run khác tương tự trong log): 94.74, 84.21, 73.68, 78.95, 89.47, 94.74, 89.47, 84.21, 94.44, 77.78 (%).

**Observations:**
- Kết hợp 3 ý tưởng **chưa** vượt baseline paper trên MUTAG; mean **thấp hơn** ~7 điểm phần trăm so với paper.
- Adaptive weights hội tụ gần tỉ lệ ~2:1:0.5 (precision L0 > L1 > L2), tương thích prior `[1, 0.5, 0.25]`.
- Biến thiên giữa các fold vẫn cao (std trong run tới ~9.5%); cần thêm run thứ 5 hoặc seed cố định để so sánh ổn định với paper.

---

### EXP 07 — HMS-JEPA Baseline, All Datasets (Kaggle H100)

**Setup:** chạy tuần tự các dataset `MUTAG, PROTEINS, IMDB-B, IMDB-M, DD, ZINC` với tối ưu H100 (`metis.online=False`, AMP bfloat16, `cudnn.benchmark=True`, batch lớn hơn). **Kaggle**, 2026-04-04.

**Config nổi bật:** `train.runs=5` cho TUD datasets (riêng ZINC dự kiến 10), `k=10` CV cho classification; `MUTAG` dùng 50 epochs, batch 512.

**Kết quả đã hoàn tất:**
- **MUTAG:** **85.68±1.50%** (5 runs x 10-fold CV), thấp hơn paper Graph-JEPA (91.25±5.75) **5.57 điểm phần trăm**.
- Wall time MUTAG: **~9.7 min**.

**Per-run MUTAG (mean acc / std over 10 folds):**

| Run | Mean acc | Std (across folds) |
|-----|----------|--------------------|
| 1   | 83.04%   | 9.89%              |
| 2   | 85.12%   | 9.98%              |
| 3   | 87.25%   | 4.16%              |
| 4   | 86.23%   | 5.29%              |
| 5   | 86.78%   | 5.37%              |
| **Mean of run means** | **85.68%** | **1.50%** (std of 5 run means) |

**Trạng thái các dataset còn lại:**
- **PROTEINS:** job bị dừng giữa run do lỗi:
  - `NameError: name 'exit' is not defined`
  - stack trace trỏ vào `core/get_data.py` nhánh `Dataset not supported.` gọi `exit(1)`.
- **IMDB-B, IMDB-M, DD, ZINC:** chưa chạy tới do crash ở PROTEINS.

**Notes nhanh để resume:**
- Kiểm tra chuỗi tên dataset trong config (`PROTEINS` vs key mà `create_dataset()` hỗ trợ).
- Sửa `exit(1)` trong `core/get_data.py` thành `sys.exit(1)` hoặc `raise ValueError(...)` để tránh lỗi NameError và có thông báo rõ ràng hơn.

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
