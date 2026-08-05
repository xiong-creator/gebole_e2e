"""PigeonEyeSpeedDataset: 加载鸽眼图像 + 距离桶 → 速度百分比目标"""
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from config import IMG_SIZE, DISTANCE_BUCKETS, INDEX_DIR


def load_speed_range(path: Path = None):
    """读取 speed_range.json，返回按 bucket_idx 索引的 (low, high) 数组"""
    path = path or (INDEX_DIR / "speed_range.json")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    ranges = []
    for name in DISTANCE_BUCKETS:
        v = raw[name]
        ranges.append((float(v["low"]), float(v["high"])))
    return ranges  # list[(low, high)] 长度=桶数


def get_transforms(train: bool):
    if train:
        return transforms.Compose(
            [
                transforms.Resize(
                    (IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(
                (IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )


class PigeonEyeSpeedDataset(Dataset):
    def __init__(self, jsonl_path: Path, speed_ranges, train: bool = False):
        self.records = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.records.append(json.loads(line))
        self.speed_ranges = speed_ranges
        self.transform = get_transforms(train=train)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        for _ in range(5):
            r = self.records[idx]
            img_path = r["image_abs_path"]
            try:
                img = Image.open(img_path).convert("RGB")
            except (FileNotFoundError, OSError):
                # 换一条继续
                idx = (idx + 1) % len(self.records)
                continue

            img_tensor = self.transform(img)
            bucket_idx = int(r["bucket_idx"])
            low, high = self.speed_ranges[bucket_idx]
            speed = float(r["speed_mpm"])
            # target percentage in [0,1]
            if high > low:
                target = (speed - low) / (high - low)
            else:
                target = 0.5
            target = max(0.0, min(1.0, target))

            return {
                "image": img_tensor,
                "bucket_idx": torch.tensor(bucket_idx, dtype=torch.long),
                "target_pct": torch.tensor(target, dtype=torch.float32),
                "speed_mpm": torch.tensor(speed, dtype=torch.float32),
                "low": torch.tensor(low, dtype=torch.float32),
                "high": torch.tensor(high, dtype=torch.float32),
            }
        raise RuntimeError(f"连续 5 条记录读取失败，起始 idx={idx}")
