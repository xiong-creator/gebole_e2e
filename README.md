# E2E 鸽眼 → 速度百分比 预测

给定 **鸽眼特写图**，分别输出 `short / middle / long` 三个距离桶下的速度预测。
模型内部先预测一个 `[0, 1]` 的百分比 `p`，最终速度 = `speed_low + p * (speed_high - speed_low)`。
其中 `[speed_low, speed_high]` 由训练集各桶速度分布统计得到，下界不再用 `0` 兜底，而是至少不低于该桶的最小正速度。

## 距离分桶

| 桶 | 范围 (km) |
|---|---|
| short  | `[1, 300)` |
| middle | `[300, 500)` |
| long   | `[500, +∞)` |

在 [config.py](file:///home/xiongyajiao/gebole/CausalFSFG_TMM/e2e/config.py) 中通过 `DISTANCE_BOUNDARIES` 调整。

## 目录结构

```
e2e/
├── config.py         # 路径、距离桶、backbone、超参
├── build_index.py    # 扫 JSONL → speed_range.json + train/val/test.jsonl
├── dataset.py        # PyTorch Dataset
├── model.py          # 冻结 DINOv2 + 可训练 Attention Probe
├── train.py          # 只训练 Probe (AMP + Cosine schedule)
├── predict.py        # 单张图片推理
├── clean_unused_images.py  # 删除 JSONL 未引用的杂图
├── index/                  # build_index 生成
└── runs/                   # 训练输出 (best.pth / last.pth / log.jsonl)
```

## 模型结构

```
image (B,3,518,518)
        │
        ▼
┌────────────────────────┐
│ DINOv2 ViT-B/14 (冻结) │  只作特征编码，永远 eval + no_grad
└────────────────────────┘
        │
        ▼  tokens (B, 1+1369, 768)  [CLS] + 37x37 patch
┌────────────────────────┐
│ Attention Probe (可训) │
│  · bucket embedding    │──┐
│    → learnable query    │  │ cross-attn
│  · N × (MHA + FFN)     │◄─┘
│  · LN + MLP head       │
│  · Sigmoid → pct ∈ (0,1)│
└────────────────────────┘
        │
        ▼
speed = low + pct × (high - low)
```

- **Backbone**：默认 `dinov2_vitb14`（86M 参数，全部冻结，不进 optimizer）
- **Probe**：距离桶 embedding + 可学习基向量 → query，与冻结 tokens 做 `MultiheadAttention`；默认 2 层 × 8 head，仅几 M 可训练参数
- 训练用 AdamW + Cosine + AMP FP16。DINOv2 前向在 `no_grad` 下执行，显存开销极低

支持的编码器（改 `config.py` 或 `--backbone`）：
`dinov2_vits14 / dinov2_vitb14 / dinov2_vitl14 / dinov2_vitg14`（及对应 `_reg` 版本）。

## 数据放置

- **JSONL 清单**：默认在 `data/E2E/zuhuanwang_race_records_20251118_20251120.with_eye.full_manifest.jsonl`
- **鸽眼图片**：放到 `data/E2E/eyes/` 目录下，**允许任意层级子目录**。
- `build_index.py` 会优先使用 JSONL 中的 `eye_closeup_first_file_path / eye_closeup_file_paths` 来定位图片；如果路径对不上，再退化到按文件名匹配。
- 训练和推理都使用固定输入尺寸，当前配置是接近 `512x512` 的合法 DINOv2 尺寸 `518x518`。

图片可以边上传边跑：`build_index.py` 会递归扫图片目录，只保留能匹配到本地图片的记录。

## 使用方式

### 一键训练

```bash
cd e2e
bash run_train.sh
# 想改超参数直接透传：
bash run_train.sh --epochs 50 --batch-size 32 --backbone dinov2_vits14
```

### 或者分两步

```bash
cd e2e
python build_index.py    # 递归扫图，生成 index/{train,val,test}.jsonl + speed_range.json
python train.py --epochs 30
```

### 推理

```bash
cd e2e

# 输入图片，输出三个距离桶下的速度预测
bash infer.sh /path/to/eye.jpg
```

## 关键设计

1. **百分比输出**：`Sigmoid` 保证 `p ∈ (0, 1)`；速度用桶内分位数区间反解 → 天然抗 OOD。
2. **DINOv2 只做编码**：整个 backbone 在 `no_grad` + `eval` 下运行，不参与训练。
3. **Attention Probe**：距离桶 embedding 生成的 query 通过 cross-attention 从 patch tokens 中"挑选"相关区域，比单纯拼接 CLS 特征更有解释性和表达力。
4. **按 ring_number 划分**：同一只鸽子的记录不会同时出现在 train/val/test，避免记忆。
5. **速度范围只用训练集统计**，避免验证/测试集泄露。
