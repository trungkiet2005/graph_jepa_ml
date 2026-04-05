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
| 01   | HMS-JEPA Baseline (reproduce)     | 86.00±1.60*      | 4/5 runs  | −5.25pp     | Kaggle 2026-04-04; batch 128, `metis.online: True`; run 4 log bị cắt |
| 02   | VICReg Regularization (C-JEPA)    | 87.09±6.51*      | 4/5 runs  | -4.16pp     | Loss scale ~4M — VICReg cov term quá mạnh |
| 03   | Layer-wise Attention Pool          | 84.56±1.38*      | 4/5 runs  | −6.69pp     | Kaggle 2026-04-04; `GraphHMSJepaLayerAttn`, `metis.online: True`; run 4 chưa có `Acc mean` |
| 04   | Multi-view Re-masking             | 85.75±0.64*      | 4/5 runs  | −5.50pp     | Kaggle 2026-04-04; `GraphHMSJepaMultiView`, K=4 views, drop 0.15; `metis.online: True`; run 4 chưa có `Acc mean` |
| 05   | Adaptive Loss Weights (Kendall)   | 86.67±1.33*      | 4/5 runs  | −4.58pp     | Kaggle 2026-04-04; `GraphHMSJepaAdaptive`, Kendall uncertainty; `metis.online: True`; run 4 chưa có `Acc mean` |
| 06   | Combined (VICReg+LayerAttn+Adapt) | 84.55±1.69*      | 4/5 runs  | −6.70pp     | Kaggle 2026-04-04; `GraphHMSJepaCombined`, VICReg trên pred + Kendall + layer-attn; `metis.online: True`; run 4 chưa có `Acc mean` |
| 07   | H100 All-datasets baseline         | 85.68±1.50       | PARTIAL   | −5.57pp     | Kaggle H100; MUTAG hoàn tất, job dừng ở PROTEINS do `NameError: exit` |
| 08   | ASAM/SAM Optimizer                 | —                | TO RUN    | —           | seeks flat minima for better generalization |
| 09   | Cosine Loss + Anneal VICReg + EMA  | —                | TO RUN    | —           | scale-invariant loss + stabilization |
| 10   | Stochastic Depth (DropPath)        | —                | TO RUN    | —           | implicit ensemble/regularization for small folds |
| 11   | Full-dim Poincaré Ball JEPA        | —                | TO RUN    | —           | 512D hyperbolic space with learnable curvature |
| 12   | Mamba-SSM Patch Encoder            | —                | TO RUN    | —           | Selective State Space for long-range structure |
| 13   | Multi-Context JEPA + NT-Xent       | —                | TO RUN    | —           | 2 contexts + auxiliary contrastive loss |
| 14   | Hybrid JEPA + MAE Dual Objective   | —                | TO RUN    | —           | Representation (JEPA) + Feature (MAE) reconstruction |

*exp02: only 4/5 runs completed. Run 4 chưa xong. Avg of runs 0-3: (88.27+86.67+85.15+88.25)/4 = 87.09%

*exp01: only 4/5 runs completed (run 4 chưa có dòng `Acc mean`). Aggregate từ runs 0–3: mean of run means **86.00%**, std **1.60%**.

*exp03: only 4/5 runs completed (run 4 không có dòng `Acc mean` trong log). Aggregate từ runs 0–3: **84.56±1.38%** (std của 4 giá trị mean theo run).

*exp04: only 4/5 runs completed (run 4 log cắt trước khi có `Acc mean`). Aggregate từ runs 0–3: **85.75±0.64%** (std của 4 giá trị mean theo run).

*exp05: only 4/5 runs completed (run 4 log cắt trước khi có `Acc mean`). Aggregate từ runs 0–3: **86.67±1.33%** (std của 4 giá trị mean theo run).

*exp06: only 4/5 runs completed (run 4 chưa có `Acc mean` trong log). Aggregate từ runs 0–3: **84.55±1.69%** (std của 4 giá trị mean theo run).

---

## Our Experiments — All Datasets (to be filled as experiments run)

| Exp  | PROTEINS | MUTAG    | DD       | REDDIT-B | REDDIT-M5 | IMDB-B   | IMDB-M   | ZINC     |
|------|----------|----------|----------|----------|-----------|----------|----------|----------|
| 01   | —        | 86.00±1.60* | —        | —        | —         | —        | —        | —        |
| 02   | —        | 87.09*   | —        | —        | —         | —        | —        | —        |
| 03   | —        | 84.56±1.38* | —        | —        | —         | —        | —        | —        |
| 04   | —        | 85.75±0.64* | —        | —        | —         | —        | —        | —        |
| 05   | —        | 86.67±1.33* | —        | —        | —         | —        | —        | —        |
| 06   | —        | 84.55±1.69* | —        | —        | —         | —        | —        | —        |
| 07   | CRASH (`NameError: exit`) | 85.68±1.50 | —        | —        | —         | —        | —        | —        |
| 08   | —        | —        | —        | —        | —         | —        | —        | —        |
| 09   | —        | —        | —        | —        | —         | —        | —        | —        |
| 10   | —        | —        | —        | —        | —         | —        | —        | —        |
| 11   | —        | —        | —        | —        | —         | —        | —        | —        |
| 12   | —        | —        | —        | —        | —         | —        | —        | —        |
| 13   | —        | —        | —        | —        | —         | —        | —        | —        |
| 14   | —        | —        | —        | —        | —         | —        | —        | —        |

---

## Detailed Run Logs

### EXP 01 — HMS-JEPA Baseline (MUTAG reproduce)

**Setup:** `exp_01_hms_jepa_mutag.py` trên **Kaggle**, 2026-04-04. Clone repo, METIS qua `libmetis-dev`, PyG wheels.

**Config (inline YAML):** `dataset: MUTAG`, Hadamard, GINEConv, hidden=512, nlayer_gnn=2, nlayer_mlpmixer=4, `n_patches: 32`, RWSE 15/15, `jepa` như baseline; `train`: epochs=50, lr=5e-4, **batch_size=128**, `runs: 5`; `k=10` (10-fold CV). Config in được merge với default → **`metis.online: True`**.

**Aggregate (mean ± std của các giá trị mean accuracy theo run, 4 run hoàn chỉnh):** **86.00±1.60%** (paper Graph-JEPA MUTAG: **91.25±5.75%**), chênh **~−5.25 điểm phần trăm** so với paper (mean).

**Per-run summary (`Acc mean` / `std` trên 10 fold):**

| Run | Mean acc | Std (across folds) |
|-----|----------|--------------------|
| 0   | 88.27%   | 8.68%              |
| 1   | 85.64%   | 10.85%             |
| 2   | 85.56%   | 7.11%              |
| 3   | 84.53%   | 7.05%              |
| 4   | —        | (log cắt trước khi có summary) |
| **Mean of run means (runs 0–3)** | **86.00%** | **1.60%** (std of 4 run means) |

**Ghi chú log:** `UserWarning` truy cập `dataset.data.y` trên `InMemoryDataset` (trainer); không làm dừng job.

---

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

### EXP 03 — Layer-Wise Attention Pooling (HISTOGRAPH-inspired) on MUTAG

**Setup:** notebook Kaggle, 2026-04-04. Clone `graph_jepa_ml`, METIS (`libmetis-dev`), PyG wheels. Model **`GraphHMSJepaLayerAttn`**: pool có trọng số attention trên output từng lớp GNN (`LayerAttentionPool`), còn lại theo HMS-JEPA baseline.

**Config (inline YAML):** `dataset: MUTAG`, Hadamard, GINEConv, hidden=512, `nlayer_gnn: 2`, `nlayer_mlpmixer: 4`, `n_patches: 32`, RWSE 15/15, JEPA multiscale như snippet; `train`: epochs=50, lr=5e-4, batch=128, `runs: 5`, `k=10`. Sau merge với default → **`metis.online: True`**.

**Aggregate (mean ± std của mean accuracy theo run, 4 run hoàn chỉnh):** **84.56±1.38%** (paper Graph-JEPA MUTAG: **91.25±5.75%**), chênh **~−6.69 điểm phần trăm** so với paper (mean).

**Per-run summary (`Acc mean` / `std` trên 10 fold):**

| Run | Mean acc | Std (across folds) |
|-----|----------|--------------------|
| 0   | 85.06%   | 5.94%              |
| 1   | 82.98%   | 7.76%              |
| 2   | 86.20%   | 7.64%              |
| 3   | 84.01%   | 7.62%              |
| 4   | —        | (log cắt trước khi có summary) |
| **Mean of run means (runs 0–3)** | **84.56%** | **1.38%** (std of 4 run means) |

**Ghi chú log:** `UserWarning` truy cập `dataset.data.y` trên `InMemoryDataset` (`core/trainer.py`); không làm dừng job.

---

### EXP 04 — Multi-View Re-Masking + Stochastic Context Augmentation on MUTAG

**Setup:** notebook Kaggle, 2026-04-04. Model **`GraphHMSJepaMultiView`**: trong training, **K=4** view dropout trên embedding context (L0/L1), `VIEW_DROP_RATE=0.15`, trung bình dự đoán qua các view; eval một view. Loss dùng `_compute_loss_hms` (SmoothL1 multiscale + `var_weight`).

**Config (inline YAML):** `dataset: MUTAG`, Hadamard, GINEConv, hidden=512, `nlayer_gnn: 2`, `nlayer_mlpmixer: 4`, `n_patches: 32`, RWSE 15/15, JEPA như snippet; `train`: epochs=50, lr=5e-4, batch=128, `runs: 5`, `k=10`. Sau merge → **`metis.online: True`**.

**Aggregate (mean ± std của mean accuracy theo run, 4 run hoàn chỉnh):** **85.75±0.64%** (paper Graph-JEPA MUTAG: **91.25±5.75%**), chênh **~−5.50 điểm phần trăm** so với paper (mean).

**Per-run summary (`Acc mean` / `std` trên 10 fold):**

| Run | Mean acc | Std (across folds) |
|-----|----------|--------------------|
| 0   | 85.12%   | 7.61%              |
| 1   | 85.64%   | 5.40%              |
| 2   | 85.61%   | 10.04%             |
| 3   | 86.64%   | 8.28%              |
| 4   | —        | (log cắt trước khi có summary) |
| **Mean of run means (runs 0–3)** | **85.75%** | **0.64%** (std of 4 run means) |

**Ghi chú log:** `UserWarning` truy cập `dataset.data.y` trên `InMemoryDataset` (`core/trainer.py`); không làm dừng job.

---

### EXP 05 — Adaptive Loss Weights (Kendall uncertainty) on MUTAG

**Setup:** HMS-JEPA với trọng số loss học theo homoscedastic uncertainty (Kendall et al., CVPR 2018): `L = 0.5 Σ exp(−log σ²_i) L_i + 0.5 Σ log σ²_i`; model **`GraphHMSJepaAdaptive`** (`log_var_L0/L1/L2` khởi tạo gần prior), `compute_loss_adaptive` dùng `SmoothL1` cho L0/L1/L2 + `var_weight` trên L0 như baseline. **Kaggle**, 2026-04-04.

**Config (inline YAML):** `dataset: MUTAG`, Hadamard, GINEConv, hidden=512, `nlayer_gnn: 2`, `nlayer_mlpmixer: 4`, `n_patches: 32`, RWSE 15/15, `jepa.loss_weights: [1.0, 0.5, 0.25]`, `var_weight: 0.01`; `train`: epochs=50, lr=5e-4, batch=128, `runs: 5`, `k=10`. Sau merge → **`metis.online: True`**.

**Aggregate (mean ± std của mean accuracy theo run, 4 run hoàn chỉnh):** **86.67±1.33%** (paper Graph-JEPA MUTAG: **91.25±5.75%**), chênh **~−4.58 điểm phần trăm** so với paper (mean).

**Per-run summary (`Acc mean` / `std` trên 10 fold):**

| Run | Mean acc | Std (across folds) |
|-----|----------|--------------------|
| 0   | 86.70%   | 9.70%              |
| 1   | 86.70%   | 6.46%              |
| 2   | 88.27%   | 8.36%              |
| 3   | 85.03%   | 9.06%              |
| 4   | —        | (log cắt trước khi có summary) |
| **Mean of run means (runs 0–3)** | **86.67%** | **1.33%** (std of 4 run means) |

**Per-fold accuracies — Run 0 only** (%): 73.68, 84.21, 73.68, 89.47, 94.74, 94.74, 89.47, 94.74, 100.00, 72.22.

**Ghi chú log:** `UserWarning` truy cập `dataset.data.y` trên `InMemoryDataset` (`core/trainer.py`); không làm dừng job.

**Observations:**
- Adaptive-only **cao hơn** EXP 06 combined (~86.7% vs ~84.6%) và **gần** EXP 02 VICReg (~87.1%); vẫn **thấp hơn** paper ~4.6 điểm phần trăm (mean).
- Log in epoch in `[Adaptive weights] L0≈1.99, L1≈1.00, L2≈0.50` (precision ≈ tỉ lệ 2:1:0.5), sát prior cố định `[1, 0.5, 0.25]` — uncertainty weighting **không** thay đổi mạnh cân bằng so với baseline.
- Std giữa các fold trong một run vẫn lớn (≈6–10%); nên hoàn thành run thứ 5 hoặc cố định seed nếu so sánh với paper.

---

### EXP 06 — Combined: VICReg + LayerAttn + Adaptive Weights (MUTAG)

**Setup:** Model **`GraphHMSJepaCombined`**: layer attention pooling trên output từng lớp GNN; trọng số loss Kendall (`get_adaptive_weights`); VICReg (variance hinge + off-diagonal cov) trên `pred_L0`, `pred_L1`, `pred_L2` với hệ số tổng `0.01 * (vicreg_L0 + vicreg_L1 + vicreg_L2)`; JEPA dùng `SmoothL1` (beta=0.5) + uncertainty `0.5*(w0*l0+w1*l1+w2*l2)+reg`. **Kaggle**, 2026-04-04 (log `Time: 2026/04/04 - 10:58`).

**Config (inline YAML):** `dataset: MUTAG`, Hadamard, GINEConv, hidden=512, `nlayer_gnn: 2`, `nlayer_mlpmixer: 4`, `n_patches: 32`, RWSE 15/15, `jepa.loss_weights: [1.0, 0.5, 0.25]`, `var_weight: 0.01` (trong cfg; VICReg là phần bổ sung trong `compute_loss_combined`); `train`: epochs=50, lr=5e-4, batch=128, `runs: 5`, `k=10`. Sau merge → **`metis.online: True`**.

**Aggregate (mean ± std của mean accuracy theo run, 4 run hoàn chỉnh):** **84.55±1.69%** (paper Graph-JEPA MUTAG: **91.25±5.75%**), chênh **~−6.70 điểm phần trăm** so với paper (mean).

**Per-run summary (`Acc mean` / `std` trên 10 fold):**

| Run | Mean acc | Std (across folds) |
|-----|----------|--------------------|
| 0   | 86.17%   | 7.23%              |
| 1   | 82.43%   | 6.83%              |
| 2   | 83.98%   | 5.51%              |
| 3   | 85.61%   | 9.47%              |
| 4   | —        | (log chưa có `Acc mean`) |
| **Mean of run means (runs 0–3)** | **84.55%** | **1.69%** (std of 4 run means) |

**Per-fold accuracies — Run 0 only** (%): 94.74, 84.21, 73.68, 78.95, 89.47, 94.74, 89.47, 84.21, 94.44, 77.78.

**Ghi chú log:** `UserWarning` truy cập `dataset.data.y` trên `InMemoryDataset` (`core/trainer.py`); không làm dừng job. Mỗi epoch train in `[Weights] L0≈1.99, L1≈1.00, L2≈0.50` (precision, drift nhẹ qua training).

**Observations:**
- Kết hợp VICReg + layer-attn + adaptive **không** cải thiện so với adaptive-only (EXP 05 ~86.7%) hay VICReg-only (EXP 02 ~87.1%); mean gộp ~**84.6%**, thấp hơn paper ~**6.7** điểm phần trăm — có thể do tương tác loss / scale VICReg trên prediction.
- Precision adaptive vẫn sát prior ~2:1:0.5 như EXP 05.
- Std giữa các fold trong một run vẫn lớn (tới ~9.5%); nên hoàn thành run thứ 5 hoặc seed cố định nếu so sánh với paper.

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

### EXP 08 — ASAM/SAM Optimizer
**Setup:** `exp_08_h100_sam_optimizer.py`. Sử dụng `ASAM` (Adaptive Sharpness-Aware Minimization) thay cho Adam để tìm flat minima.

---

### EXP 09 — Cosine Loss + Annealed VICReg + Cosine EMA Schedule
**Setup:** `exp_09_h100_cosine_loss.py`. Thay `SmoothL1` bằng **Cosine Similarity**. Anneal `vic_weight` từ 0.05 về 0. EMA momentum schedule 0.996 → 1.0.

---

### EXP 10 — Stochastic Depth (DropPath)
**Setup:** `exp_10_h100_stochastic_depth.py`. Thêm `DropPath` vào các lớp GNN residual and Mixer blocks.

---

### EXP 11 — Full-dimensional Poincaré Ball JEPA
**Setup:** `exp_11_h100_poincare_ball.py`. Chuyển từ 2D Lorentzian hyperbola sang **512D Poincaré ball** với learnable curvature `c`.

---

### EXP 12 — Mamba-SSM Patch Encoder
**Setup:** `exp_12_h100_mamba_ssm.py`. Thay thế Transformer Attention bằng **Selective State Space (S6/Mamba)** blocks.

---

### EXP 13 — Multi-Context Multi-Target (MCMT) JEPA + NT-Xent
**Setup:** `exp_13_h100_mcmt_jepa.py`. Sử dụng **2 context patches** predict targets + NT-Xent auxiliary contrastive loss.

---

### EXP 14 — Hybrid JEPA + MAE Dual Objective
**Setup:** `exp_14_h100_mae_jepa.py`. Kết hợp **Predictive (JEPA)** và **Generative (MAE)** reconstruction targets.

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
