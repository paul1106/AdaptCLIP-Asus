# Prediction & Metrics Workflow

將 inference 與 metrics 計算拆成兩個獨立步驟，讓你可以在不重跑模型的情況下，用不同的切割方式重新計算 metrics。

```
test.py --save-predictions          →   prediction_manifest.jsonl
                                         pixel_maps/*.npy
                                         gt_masks/*.npy
                                              ↓
                              compute_metrics.py
                                              ↓
                                     metrics.csv / metrics.json
```

---

## Step 1：跑 Inference，儲存預測結果

加上 `--save-predictions` flag，inference 結束後會直接存檔並跳過 metrics 計算。

```bash
python test.py \
    --dataset visa \
    --test_data_path ./dataset/visa \
    --checkpoint_path ./adaptclip_checkpoints/epoch_15_mvtec.pth \
    --save_path ./results/ \
    --k_shots 1 --seed 10 \
    --features_list 6 12 18 24 --image_size 518 \
    --n_ctx 12 --vl_reduction 4 --pq_mid_dim 128 \
    --visual_learner --textual_learner --pq_learner --pq_context \
    --save-predictions
```

輸出結構：

```
results/
  visa_10seed_1shot_predictions/
    prediction_manifest.jsonl     ← metadata（每行一筆 JSON）
    pixel_maps/
      00000000.npy                ← pixel anomaly map，float32 [H, W]
      00000001.npy
      ...
    gt_masks/
      00000000.npy                ← 二值化 GT mask，uint8 [H, W]
      00000001.npy
      ...
```

`prediction_manifest.jsonl` 每行格式：

```json
{
  "index": 0,
  "sample_id": "...",
  "cls_name": "bottle",
  "specie_name": "broken_large",
  "anomaly": 1,
  "img_path": "/data/visa/bottle/test/broken_large/000.png",
  "pred_score": 0.873,
  "pixel_map_path": "pixel_maps/00000000.npy",
  "gt_mask_path": "gt_masks/00000000.npy",
  "pixel_map_shape": [37, 37]
}
```

---

## Step 2：計算 Metrics

### 基本用法（按 class 分組）

```bash
python compute_metrics.py \
    --manifest results/visa_10seed_1shot_predictions/prediction_manifest.jsonl \
    --output-dir results/visa_10seed_1shot_eval/
```

### 使用自訂 CSV 過濾樣本

CSV 需包含 `img_path` 欄位（可用 `--split-col` 指定其他欄位名稱）：

```bash
python compute_metrics.py \
    --manifest .../prediction_manifest.jsonl \
    --output-dir .../eval_split_A/ \
    --split-csv splits/split_A.csv \
    --split-col img_path
```

換不同 split 只需換 `--split-csv`，不用重跑 inference：

```bash
python compute_metrics.py --manifest ... --output-dir .../eval_split_B/ --split-csv splits/split_B.csv
python compute_metrics.py --manifest ... --output-dir .../eval_split_C/ --split-csv splits/split_C.csv
```

### 選擇分組方式

| `--groupings` 選項 | 說明 |
|---|---|
| `overall` | 所有樣本合在一起算 |
| `cls_name` | 按物件類別分組（預設） |
| `specie_name` | 按缺陷類型分組（含對應類別的正常樣本）|
| `cls_name+specie_name` | 按 class × 缺陷類型 細分 |

```bash
# 只看整體和 class 層級
python compute_metrics.py ... --groupings overall cls_name

# 加上缺陷類型細分
python compute_metrics.py ... --groupings overall cls_name specie_name cls_name+specie_name
```

### 選擇 Metrics

```bash
# 預設：所有 metrics
python compute_metrics.py ... --metrics I-AUROC I-AP I-F1max P-AUROC P-AP P-F1max P-AUPRO

# 只算 image-level
python compute_metrics.py ... --metrics I-AUROC I-AP I-F1max

# 只算 pixel-level
python compute_metrics.py ... --metrics P-AUROC P-AP P-F1max P-AUPRO
```

### 大資料集（OOM 處理）

調小 `--batch-size` 降低讀取 pixel map 時的 GPU 記憶體峰值：

```bash
python compute_metrics.py ... --batch-size 8
```

如果 AUPRO 還是 OOM，加上 `--aupro-num-thresholds` 限制 threshold 採樣數：

```bash
python compute_metrics.py ... --aupro-num-thresholds 200
```

### 使用 CPU

```bash
python compute_metrics.py ... --device cpu
```

---

## 完整參數列表

| 參數 | 預設值 | 說明 |
|---|---|---|
| `--manifest` | 必填 | `prediction_manifest.jsonl` 路徑 |
| `--output-dir` | 必填 | 結果輸出目錄 |
| `--split-csv` | 無 | 過濾用的 CSV 檔案 |
| `--split-col` | `img_path` | CSV 中對應圖片路徑的欄位名稱 |
| `--groupings` | `overall cls_name` | 分組方式（可多選） |
| `--metrics` | 全部 7 個 | 要計算的 metrics（可多選） |
| `--batch-size` | `32` | 讀取 pixel map 的 batch size |
| `--device` | `cuda`（若可用）| 計算裝置 |
| `--aupro-num-thresholds` | 無（精確模式）| 固定 AUPRO threshold 數量以節省 VRAM |

---

## 輸出格式

```
output-dir/
  metrics.csv     ← 每行一個 group，可直接用 Excel 或 pandas 讀取
  metrics.json    ← 完整結構，包含 sample_count / anomaly_count 等統計
```

`metrics.csv` 範例：

```
group,sample_count,anomaly_count,normal_count,I-AUROC,I-AP,I-F1max,P-AUROC,P-AP,P-F1max,P-AUPRO
overall,1000,500,500,0.921,0.934,0.871,0.963,0.712,0.654,0.881
bottle,80,40,40,0.945,0.962,0.901,...
```

結果為 0~1 的小數，乘以 100 即為百分比。
