"""推理脚本：
1. 输入一张鸽眼图片 + 距离（公里），自动分桶并输出速度；
2. 或仅输入一张鸽眼图片，输出三个距离桶下的速度预测。
"""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from config import (
    BACKBONE,
    DISTANCE_BOUNDARY_PATH,
    DISTANCE_BUCKETS,
    DISTANCE_BOUNDARIES,
    INDEX_DIR,
    distance_to_bucket_idx_with_boundaries,
)
from dataset import get_transforms, load_speed_range
from model import EyeSpeedNet


def resolve_metadata_path(ckpt_path: Path, filename: str, fallback: Path):
    if fallback.exists():
        return fallback
    candidate = ckpt_path.parent / filename
    if candidate.exists():
        return candidate
    return fallback


def load_distance_boundaries(path: Path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return [float(v) for v in raw["boundaries"]]
    return DISTANCE_BOUNDARIES


def load_model(ckpt_path: Path, backbone: str, device: torch.device):
    model = EyeSpeedNet(backbone=backbone).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def prepare_input(image_path: Path, device: torch.device):
    tf = get_transforms(train=False)
    img = Image.open(image_path).convert("RGB")
    return tf(img).unsqueeze(0).to(device)


def predict_for_bucket(model, img_t, bucket_idx: int, ranges, boundaries):
    low, high = ranges[bucket_idx]
    bucket_t = torch.tensor([bucket_idx], dtype=torch.long, device=img_t.device)
    with torch.no_grad():
        pct = model(img_t, bucket_t).item()
    speed = low + pct * (high - low)
    return {
        "bucket": DISTANCE_BUCKETS[bucket_idx],
        "bucket_idx": bucket_idx,
        "distance_boundaries_km": boundaries,
        "speed_range_mpm": [low, high],
        "predicted_pct": pct,
        "predicted_speed_mpm": speed,
        "predicted_speed_kmh": speed * 60 / 1000,
    }


def predict(image_path: Path, distance_km: float, ckpt_path: Path,
            backbone: str = BACKBONE):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    speed_range_path = resolve_metadata_path(
        ckpt_path, "speed_range.json", INDEX_DIR / "speed_range.json"
    )
    boundary_path = resolve_metadata_path(
        ckpt_path, "distance_boundaries.json", DISTANCE_BOUNDARY_PATH
    )
    ranges = load_speed_range(speed_range_path)
    boundaries = load_distance_boundaries(boundary_path)

    model = load_model(ckpt_path, backbone, device)
    img_t = prepare_input(image_path, device)

    bucket_idx = distance_to_bucket_idx_with_boundaries(distance_km, boundaries)
    pred = predict_for_bucket(model, img_t, bucket_idx, ranges, boundaries)

    return {
        "image": str(image_path),
        "distance_km": distance_km,
        **pred,
    }


def predict_all_buckets(image_path: Path, ckpt_path: Path, backbone: str = BACKBONE):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    speed_range_path = resolve_metadata_path(
        ckpt_path, "speed_range.json", INDEX_DIR / "speed_range.json"
    )
    boundary_path = resolve_metadata_path(
        ckpt_path, "distance_boundaries.json", DISTANCE_BOUNDARY_PATH
    )
    ranges = load_speed_range(speed_range_path)
    boundaries = load_distance_boundaries(boundary_path)

    model = load_model(ckpt_path, backbone, device)
    img_t = prepare_input(image_path, device)

    preds = []
    for bucket_idx in range(len(DISTANCE_BUCKETS)):
        preds.append(predict_for_bucket(model, img_t, bucket_idx, ranges, boundaries))
    return {
        "image": str(image_path),
        "distance_boundaries_km": boundaries,
        "predictions": preds,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, required=True)
    ap.add_argument("--distance-km", type=float, default=None)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--backbone", type=str, default=BACKBONE,
                    help="需要与训练时保持一致")
    ap.add_argument("--all-buckets", action="store_true",
                    help="忽略输入距离，直接输出所有距离桶下的预测")
    ap.add_argument("--json", action="store_true", help="以 JSON 形式输出")
    args = ap.parse_args()

    if args.distance_km is None and not args.all_buckets:
        args.all_buckets = True

    if args.all_buckets:
        out = predict_all_buckets(Path(args.image), Path(args.ckpt),
                                  backbone=args.backbone)
    else:
        out = predict(Path(args.image), args.distance_km, Path(args.ckpt),
                      backbone=args.backbone)

    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    if args.all_buckets:
        print(f"image: {out['image']}")
        print(f"distance_boundaries_km: {out['distance_boundaries_km']}")
        for item in out["predictions"]:
            print("-" * 40)
            for k, v in item.items():
                print(f"{k}: {v}")
        return

    for k, v in out.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
