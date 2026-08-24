"""端到端 鸽眼 + 距离 → 速度百分比 训练/推理 配置"""
from pathlib import Path

# ============ 路径 ============
PROJECT_ROOT = Path("~/gebole/CausalFSFG_TMM/e2e").expanduser()
DATA_DIR = PROJECT_ROOT / "data" / "E2E"

# JSONL 数据清单
# 允许同时使用多份 manifest 共同建索引与训练。
# 默认第一份是历史主清单，第二份可替换成最新 streamed+score 清单。
JSONL_PATHS = [
    DATA_DIR / "zuhuanwang_race_records_20251118_20251120.with_eye.full_manifest.jsonl",
    DATA_DIR / "raw_video" / "raw_video.streamed.cc518.with_scores.20260824_072129.jsonl",
]

# 兼容旧脚本：保留单路径别名，默认指向第一份清单。
JSONL_PATH = JSONL_PATHS[0]

# 鸽眼图像根目录：
# 当前数据放在 data/E2E/eyes 下，允许任意层级子目录。
# 优先使用 JSONL 中的 `eye_closeup_first_file_path / eye_closeup_file_paths`
# 来定位文件；必要时再退化到按 basename 匹配。
IMAGE_ROOT = DATA_DIR / "eyes"

# 输出目录
OUT_DIR = PROJECT_ROOT / "runs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR = PROJECT_ROOT / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# ============ 距离分桶 ============
# 单位：公里
# 固定业务分桶：
# b0: distance < 200
# b1: 200 <= distance < 300
# b2: 300 <= distance < 400
# b3: 400 <= distance < 500
# b4: distance >= 500
DISTANCE_BUCKETS = ["lt200", "200_300", "300_400", "400_500", "ge500"]
DISTANCE_BOUNDARIES = [200.0, 300.0, 400.0, 500.0]
DISTANCE_BOUNDARY_PATH = INDEX_DIR / "distance_boundaries.json"


def distance_to_bucket_idx(distance_km: float) -> int:
    if distance_km < DISTANCE_BOUNDARIES[0]:
        return 0
    elif distance_km < DISTANCE_BOUNDARIES[1]:
        return 1
    elif distance_km < DISTANCE_BOUNDARIES[2]:
        return 2
    elif distance_km < DISTANCE_BOUNDARIES[3]:
        return 3
    else:
        return 4


def distance_to_bucket_idx_with_boundaries(distance_km: float, boundaries) -> int:
    for idx, boundary in enumerate(boundaries):
        if distance_km < boundary:
            return idx
    return len(boundaries)


# ============ 训练超参 ============
# 注意：DINOv2 的 patch size 是 14，图像边必须是 14 的倍数
# 512 不是 14 的倍数，因此固定到最接近的合法尺寸 518 (= 14 * 37)
# 以满足“训练/推理固定为 512 级别输入”的需求，同时保证 DINOv2 正常工作。
IMG_SIZE = 518
BATCH_SIZE = 64
NUM_WORKERS = 8
EPOCHS = 30
# 只训 probe，单一学习率即可
LR = 3e-4
WEIGHT_DECAY = 1e-4

# 冻结的 DINOv2 编码器（只作特征提取，不参与训练）
#   dinov2_vits14 / dinov2_vitb14 / dinov2_vitl14 / dinov2_vitg14
#   (以及对应的 _reg 版本)
BACKBONE = "dinov2_vitb14"

# Attention Probe 结构
PROBE_HEADS = 8
PROBE_LAYERS = 2
PROBE_MLP_RATIO = 2.0
PROBE_DROPOUT = 0.1

# 用于速度范围计算的分位数（去极端值）
SPEED_QUANTILE_LOW = 0.01
SPEED_QUANTILE_HIGH = 0.99
SPEED_RANGE_EXPAND_LOW = 20.0
SPEED_RANGE_EXPAND_HIGH = 20.0

# 训练样本速度异常值过滤：
# 先按距离桶统计速度分布，再用分位数区间过滤明显异常值，
# 防止少量脏标签同时污染训练样本和 speed_range 上下限。
SPEED_FILTER_ENABLED = True
SPEED_FILTER_QUANTILE_LOW = 0.01
SPEED_FILTER_QUANTILE_HIGH = 0.99
SPEED_FILTER_EXPAND_LOW = 0.0
SPEED_FILTER_EXPAND_HIGH = 0.0

# 数据划分随机种子
SEED = 42
VAL_RATIO = 0.05
TEST_RATIO = 0.05
