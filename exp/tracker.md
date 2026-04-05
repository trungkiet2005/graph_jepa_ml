# Experiment Tracker — Graph-JEPA Improvements

> Goal: Beat the original Graph-JEPA paper and target Oral at A* conference.
> All experiments use **10-fold CV**, reported as **mean +/- std over 5 runs** (matching the paper's protocol).
> Metric: Classification Accuracy (%) for TUD datasets, MAE for ZINC.

---

## Table 1 — Paper baselines + our experiments (unified)

Cùng một bảng để so **trực tiếp** với Table 1 trong paper (Skenderi et al.; bản gốc xem `Original_Paper.md` / arXiv:2309.16014). Hàng **Graph-JEPA (paper)** là baseline gốc; các hàng **Ours — Exp …** là chạy trong repo này (cùng cột, cùng metric). **Δ vs paper:** với classification, lấy *mean* của mình trừ *mean* của paper (cùng dataset) → điểm phần trăm (pp); với ZINC (MAE), so sánh số MAE (thấp hơn = tốt hơn). Ô **—** = chưa chạy / không có trong paper.

| Method | PROTEINS | MUTAG | DD | REDDIT-B | REDDIT-M5 | IMDB-B | IMDB-M | ZINC (MAE) |
|--------|----------|-------|-----|----------|-----------|--------|--------|------------|
| F-GIN (supervised) | 72.39±2.76 | 90.41±4.61 | 74.87±3.56 | 86.79±2.04 | 53.28±3.17 | 71.83±1.93 | 48.46±2.31 | 0.254±0.005 |
| InfoGraph | 72.57±0.65 | 87.71±1.77 | 75.23±0.39 | 78.79±2.14 | 51.11±0.55 | 71.11±0.88 | 48.66±0.67 | 0.890±0.017 |
| GraphCL | 72.86±1.01 | 88.29±1.31 | 74.70±0.70 | 82.63±0.99 | 53.05±0.40 | 70.80±0.77 | 48.49±0.63 | 0.627±0.013 |
| MVGRL | — | — | — | 84.5±0.6 | — | 74.2±0.7 | 51.2±0.5 | — |
| AD-GCL-FIX | 73.59±0.65 | 89.25±1.45 | 74.49±0.52 | 85.52±0.79 | 53.00±0.82 | 71.57±1.01 | 49.04±0.53 | 0.578±0.012 |
| AD-GCL-OPT | 73.81±0.46 | 89.70±1.03 | 75.10±0.39 | 85.52±0.79 | 54.93±0.43 | 72.33±0.56 | 49.89±0.66 | 0.544±0.004 |
| GraphMAE | 75.30±0.39 | 88.19±1.26 | 74.27±1.07 | 88.01±0.19 | 46.06±3.44 | 75.52±0.66 | 51.63±0.52 | 0.935±0.034 |
| S2GAE | 76.37±0.43 | 88.26±0.76 | — | 87.83±0.27 | — | 75.76±0.62 | 51.79±0.36 | — |
| LaGraph | 75.2±0.4 | 90.2±1.1 | 78.1±0.4 | 90.4±0.8 | 56.4±0.4 | 73.7±0.9 | — | — |
| **Graph-JEPA (paper)** | **75.68±3.78** | **91.25±5.75** | **78.64±2.35** | **91.99±1.59** | **56.73±1.96** | **73.68±3.24** | **50.69±2.91** | **0.434±0.014** |
| Ours — Exp 07 (H100 all-datasets baseline) | 73.14±0.52 | 87.96±1.11 | 77.11±0.45 | — | — | 72.86±0.37 | 50.93±0.23 | 0.4503±0.0101 |
| Ours — Exp 09 (Cosine + anneal VICReg + EMA) | 73.47±1.05 | 87.42±0.93 | 76.57±0.38 | — | — | 73.42±0.26 | 50.64±0.36 | — |

*(Các exp chỉ MUTAG hoặc chưa chạy đủ suite: xem bảng «Our Experiments — MUTAG» và bảng index Exp 01–14 bên dưới.)*

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
| 07   | H100 All-datasets baseline         | 87.96±1.11       | ✅ DONE    | −3.29pp     | Kaggle H100; all 6 datasets complete, 119.1 min total wall time |
| 08   | ASAM/SAM Optimizer                 | —                | TO RUN    | —           | seeks flat minima for better generalization |
| 09   | Cosine Loss + Anneal VICReg + EMA  | 87.42±0.93       | ✅ DONE    | −3.83pp     | Kaggle H100 2026-04-05; cosine pred + VIC anneal 0.05→0 + EMA 0.996→1; **no ZINC** in script |
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

## Our Experiments — All Datasets (index by Exp ID)

**So với paper:** Hàng **Exp 07 / Exp 09** (đủ cột đã chạy) nằm trong **Table 1 — Paper baselines + our experiments (unified)** phía trên, cạnh Graph-JEPA (paper) và các baseline. Bảng dưới là **lưới Exp 01–14** theo ID; số trùng với hàng «Ours» trong Table 1 unified.

**vs paper / “leaderboard” row:** Mỗi ô so với **Graph-JEPA (paper)** cùng cột trong Table 1 unified. (Bảng MUTAG-only: **vs Paper** chỉ trên MUTAG.)

| Exp  | PROTEINS | MUTAG    | DD       | REDDIT-B | REDDIT-M5 | IMDB-B   | IMDB-M   | ZINC     |
|------|----------|----------|----------|----------|-----------|----------|----------|----------|
| 01   | —        | 86.00±1.60* | —        | —        | —         | —        | —        | —        |
| 02   | —        | 87.09*   | —        | —        | —         | —        | —        | —        |
| 03   | —        | 84.56±1.38* | —        | —        | —         | —        | —        | —        |
| 04   | —        | 85.75±0.64* | —        | —        | —         | —        | —        | —        |
| 05   | —        | 86.67±1.33* | —        | —        | —         | —        | —        | —        |
| 06   | —        | 84.55±1.69* | —        | —        | —         | —        | —        | —        |
| 07   | 73.14±0.52 | 87.96±1.11 | 77.11±0.45 | —        | —         | 72.86±0.37 | 50.93±0.23 | 0.4503±0.0101 |
| 08   | —        | —        | —        | —        | —         | —        | —        | —        |
| 09   | 73.47±1.05 | 87.42±0.93 | 76.57±0.38 | —        | —         | 73.42±0.26 | 50.64±0.36 | —        |
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

**Setup:** chạy tuần tự các dataset `MUTAG, PROTEINS, IMDB-B, IMDB-M, DD, ZINC` với tối ưu H100 (`metis.online=False`, AMP bfloat16, `cudnn.benchmark=True`, batch lớn hơn). **Kaggle**, 2026-04-05.

**Config nổi bật:** `train.runs=5` cho TUD datasets, `train.runs=10` cho ZINC, `k=10` CV cho classification. Batch sizes: MUTAG/PROTEINS/IMDB-B/IMDB-M/ZINC=512, DD=256.

**Kết quả tổng hợp (all 6 datasets complete ✅):**

| Dataset    | Result            | Paper              | Δ vs Paper  | Wall time |
|------------|-------------------|--------------------|-------------|----------|
| MUTAG      | **87.96±1.11%**   | 91.25±5.75%        | −3.29pp     | 9.2 min  |
| PROTEINS   | **73.14±0.52%**   | 75.68±3.78%        | −2.54pp     | 23.7 min |
| IMDB-B     | **72.86±0.37%**   | 73.68±3.24%        | −0.82pp     | 12.5 min |
| IMDB-M     | **50.93±0.23%**   | 50.69±2.91%        | **+0.24pp** | 25.0 min |
| DD         | **77.11±0.45%**   | 78.64±2.35%        | −1.53pp     | 32.7 min |
| ZINC (MAE) | **0.4503±0.0101** | 0.434±0.014        | +0.016      | 16.1 min |
| **Total**  |                   |                    |             | **119.1 min** |

**Per-run MUTAG (mean acc / std over 10 folds):**

| Run | Mean acc | Std (across folds) |
|-----|----------|--------------------|
| 1   | 88.30%   | 5.20%              |
| 2   | 86.67%   | 4.44%              |
| 3   | 86.67%   | 7.40%              |
| 4   | 88.83%   | 7.40%              |
| 5   | 89.36%   | 7.24%              |
| **Mean of run means** | **87.96%** | **1.11%** (std of 5 run means) |

**Per-run PROTEINS (mean acc / std over 10 folds):**

| Run | Mean acc | Std (across folds) |
|-----|----------|--------------------|
| 1   | 73.14%   | 4.51%              |
| 2   | 73.86%   | 1.98%              |
| 3   | 72.34%   | 4.40%              |
| 4   | 72.87%   | 2.93%              |
| 5   | 73.50%   | 5.30%              |
| **Mean of run means** | **73.14%** | **0.52%** (std of 5 run means) |

**Per-run IMDB-BINARY (mean acc / std over 10 folds):**

| Run | Mean acc | Std (across folds) |
|-----|----------|--------------------|
| 1   | 72.20%   | 5.36%              |
| 2   | 73.00%   | 3.82%              |
| 3   | 73.30%   | 4.29%              |
| 4   | 73.00%   | 3.87%              |
| 5   | 72.80%   | 4.14%              |
| **Mean of run means** | **72.86%** | **0.37%** (std of 5 run means) |

**Per-run IMDB-MULTI (mean acc / std over 10 folds):**

| Run | Mean acc | Std (across folds) |
|-----|----------|--------------------|
| 1   | 50.73%   | 2.48%              |
| 2   | 50.93%   | 3.12%              |
| 3   | 50.67%   | 3.25%              |
| 4   | 51.00%   | 2.83%              |
| 5   | 51.33%   | 3.70%              |
| **Mean of run means** | **50.93%** | **0.23%** (std of 5 run means) |

**Per-run DD (mean acc / std over 10 folds):**

| Run | Mean acc | Std (across folds) |
|-----|----------|--------------------|
| 1   | 77.33%   | 2.19%              |
| 2   | 76.23%   | 2.57%              |
| 3   | 77.16%   | 2.21%              |
| 4   | 77.42%   | 1.39%              |
| 5   | 77.42%   | 2.20%              |
| **Mean of run means** | **77.11%** | **0.45%** (std of 5 run means) |

**Per-run ZINC (MAE, 10 runs):**

| Run | MAE    |
|-----|--------|
| 1   | 0.4628 |
| 2   | 0.4562 |
| 3   | 0.4368 |
| 4   | 0.4301 |
| 5   | 0.4525 |
| 6   | 0.4539 |
| 7   | 0.4519 |
| 8   | 0.4457 |
| 9   | 0.4495 |
| 10  | 0.4639 |
| **Mean** | **0.4503±0.0101** |

**Observations:**
- IMDB-MULTI **vượt paper** (+0.24pp) — lần đầu beat paper trên 1 dataset.
- IMDB-B gần sát paper (−0.82pp) — trong khoảng dao động std của paper (3.24%).
- DD gần sát paper (−1.53pp) — trong khoảng dao động std của paper (2.35%).
- MUTAG gap thu hẹp đáng kể so với EXP 01-06 (−3.29pp vs −5.25pp trước đó) nhờ `metis.online=False` + batch lớn hơn.
- ZINC MAE 0.4503 gần paper (0.434), chênh +3.7% tương đối.
- Tổng wall time 119.1 min trên H100 — rất hiệu quả cho 6 datasets.

---

### EXP 08 — ASAM/SAM Optimizer
**Setup:** `exp_08_h100_sam_optimizer.py`. Sử dụng `ASAM` (Adaptive Sharpness-Aware Minimization) thay cho Adam để tìm flat minima.

---

### EXP 09 — Cosine Loss + Annealed VICReg + Cosine EMA Schedule

**Setup:** `exp_09_h100_cosine_loss.py` trên **Kaggle H100**, 2026-04-05. Clone repo, METIS, PyG wheels. Loss: **negative cosine similarity** (per-target-patch mean) thay `SmoothL1`; **VICReg-lite** variance hinge với trọng số **anneal cosine** 0.05 → 0; **EMA momentum** cosine 0.996 → 1.0; **grad clip** 1.0; AMP bfloat16. Datasets: **MUTAG, PROTEINS, IMDB-BINARY, IMDB-MULTI, DD** (không ZINC). `metis.online=False`, `k=10`, 5 runs × 10-fold CV.

**Kết quả tổng hợp (5 datasets ✅):**

| Dataset      | EXP 09            | Paper              | Δ vs Paper | EXP 07      | Δ vs 07   | Wall time |
|--------------|-------------------|--------------------|------------|-------------|-----------|-----------|
| MUTAG        | **87.42±0.93%**   | 91.25±5.75%        | −3.83pp    | 87.96±1.11% | −0.54pp   | 9.1 min   |
| PROTEINS     | **73.47±1.05%**   | 75.68±3.78%        | −2.21pp    | 73.14±0.52% | **+0.33pp** | 23.8 min |
| IMDB-BINARY  | **73.42±0.26%**   | 73.68±3.24%        | −0.26pp    | 72.86±0.37% | **+0.56pp** | 12.8 min |
| IMDB-MULTI   | **50.64±0.36%**   | 50.69±2.91%        | −0.05pp    | 50.93±0.23% | −0.29pp   | 24.2 min |
| DD           | **76.57±0.38%**   | 78.64±2.35%        | −2.07pp    | 77.11±0.45% | −0.54pp   | 33.1 min |
| **Total**    |                   |                    |            |             |           | **103.1 min** |

**Tracker lines (copy-paste):**

- `[TRACKER] MUTAG: 87.42+/-0.93%` — Train Loss: 0.0960 ± 0.0125; ~0.19 s/epoch
- `[TRACKER] PROTEINS: 73.47+/-1.05%` — Train Loss: 0.1145 ± 0.0161; ~0.49 s/epoch
- `[TRACKER] IMDB-BINARY: 73.42+/-0.26%` — Train Loss: 0.2073 ± 0.0181; ~0.52 s/epoch
- `[TRACKER] IMDB-MULTI: 50.64+/-0.36%` — Train Loss: 0.2182 ± 0.0163; ~0.62 s/epoch
- `[TRACKER] DD: 76.57+/-0.38%` — Train Loss: 0.1057 ± 0.0276; ~0.94 s/epoch

**Per-run means (FINAL RESULTS blocks):**

| Dataset     | Run means (%) |
|-------------|----------------|
| MUTAG       | 86.67, 88.80, 87.72, 87.75, 86.14 → **87.42±0.93** |
| PROTEINS    | 72.42, 74.13, 71.98, 74.58, 74.22 → **73.47±1.05** |
| IMDB-BINARY | 73.00, 73.70, 73.40, 73.70, 73.30 → **73.42±0.26** |
| IMDB-MULTI  | 50.53, 50.07, 51.13, 50.87, 50.60 → **50.64±0.36** |
| DD          | 76.82, 76.57, 76.40, 75.97, 77.08 → **76.57±0.38** |

**Observations:**
- **PROTEINS** và **IMDB-BINARY** nhích **cao hơn EXP 07** (+0.33pp / +0.56pp); MUTAG, IMDB-M, DD **thấp hơn** nhẹ so với 07.
- IMDB-MULTI gần **sát paper** (−0.05pp) — trong nhiễu thống kê của paper.
- Tổng wall **103.1 min** (ước ~65 min trong comment script — thực tế dài hơn, có thể do I/O Kaggle / log).

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
