"""Evaluate a checkpoint on val/test and generate detailed visualizations."""
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from config import BACKBONE, DISTANCE_BUCKETS, INDEX_DIR, OUT_DIR
from dataset import get_transforms, load_speed_range
from model import EyeSpeedNet


EVAL_SPLIT = "test"
RUN_DIR = Path("/home/xiongyajiao/gebole/CausalFSFG_TMM/e2e/runs/run_20260824_074343")
CKPT_PATH = RUN_DIR / "best.pth"
OUTPUT_DIR = Path(__file__).resolve().parent / "figures" / "model_eval_new"
_BUCKET_COLOR_CYCLE = [
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#B279A2",
]
BUCKET_COLORS = {
    name: _BUCKET_COLOR_CYCLE[idx % len(_BUCKET_COLOR_CYCLE)]
    for idx, name in enumerate(DISTANCE_BUCKETS)
}
ERROR_THRESHOLDS = [50.0, 100.0, 200.0]
BATCH_SIZE = 8


class EvalDataset(Dataset):
    def __init__(self, split_path: Path):
        self.records = []
        with open(split_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                row["bucket_name"] = DISTANCE_BUCKETS[int(row["bucket_idx"])]
                self.records.append(row)
        self.transform = get_transforms(train=False)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records[idx]
        image = Image.open(row["image_abs_path"]).convert("RGB")
        return {
            "image": self.transform(image),
            "bucket_idx": int(row["bucket_idx"]),
            "bucket_name": row["bucket_name"],
            "distance_km": float(row["distance_km"]),
            "speed_mpm": float(row["speed_mpm"]),
            "ring_number": row["ring_number"],
            "image_file_name": row["image_file_name"],
            "race_name": row.get("race_name") or "",
        }


def collate_fn(batch):
    return {
        "image": torch.stack([item["image"] for item in batch], dim=0),
        "bucket_idx": torch.tensor([item["bucket_idx"] for item in batch], dtype=torch.long),
        "bucket_name": [item["bucket_name"] for item in batch],
        "distance_km": [item["distance_km"] for item in batch],
        "speed_mpm": [item["speed_mpm"] for item in batch],
        "ring_number": [item["ring_number"] for item in batch],
        "image_file_name": [item["image_file_name"] for item in batch],
        "race_name": [item["race_name"] for item in batch],
    }


def resolve_speed_range_path():
    candidate = RUN_DIR / "speed_range.json"
    return candidate if candidate.exists() else INDEX_DIR / "speed_range.json"


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EyeSpeedNet(backbone=BACKBONE).to(device)
    state = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, device


def decode_speed(pct, low, high):
    return low + pct * (high - low)


def evaluate_split(split_name: str):
    split_path = INDEX_DIR / f"{split_name}.jsonl"
    dataset = EvalDataset(split_path)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn)
    ranges = load_speed_range(resolve_speed_range_path())
    model, device = load_model()

    predictions = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            image = batch["image"].to(device)
            bucket_idx = batch["bucket_idx"].to(device)
            pct = model(image, bucket_idx).cpu().tolist()
            for i, bucket in enumerate(batch["bucket_idx"].tolist()):
                low, high = ranges[bucket]
                pred_speed = decode_speed(pct[i], low, high)
                gt_speed = batch["speed_mpm"][i]
                error = pred_speed - gt_speed
                predictions.append(
                    {
                        "split": split_name,
                        "ring_number": batch["ring_number"][i],
                        "image_file_name": batch["image_file_name"][i],
                        "race_name": batch["race_name"][i],
                        "distance_km": batch["distance_km"][i],
                        "bucket_idx": bucket,
                        "bucket_name": DISTANCE_BUCKETS[bucket],
                        "gt_speed_mpm": gt_speed,
                        "pred_speed_mpm": pred_speed,
                        "abs_error_mpm": abs(error),
                        "signed_error_mpm": error,
                        "pred_pct": pct[i],
                    }
                )
            if batch_idx % 20 == 0:
                print(f"[eval] processed {min(batch_idx * BATCH_SIZE, len(dataset))}/{len(dataset)}")
    return predictions


def regression_metrics(rows):
    n = len(rows)
    gt = [row["gt_speed_mpm"] for row in rows]
    pred = [row["pred_speed_mpm"] for row in rows]
    abs_err = [abs(row["signed_error_mpm"]) for row in rows]
    sq_err = [(row["signed_error_mpm"]) ** 2 for row in rows]
    mean_gt = sum(gt) / n
    ss_tot = sum((v - mean_gt) ** 2 for v in gt)
    ss_res = sum((p - g) ** 2 for p, g in zip(pred, gt))
    mape_base = [abs((p - g) / g) for p, g in zip(pred, gt) if g > 1e-6]
    metrics = {
        "count": n,
        "mae_mpm": sum(abs_err) / n,
        "rmse_mpm": math.sqrt(sum(sq_err) / n),
        "mape": sum(mape_base) / max(1, len(mape_base)),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0,
    }
    for thr in ERROR_THRESHOLDS:
        metrics[f"acc_within_{int(thr)}mpm"] = sum(err <= thr for err in abs_err) / n
    return metrics


def save_predictions_csv(rows, path: Path):
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_metrics(rows):
    by_bucket = defaultdict(list)
    for row in rows:
        by_bucket[row["bucket_name"]].append(row)

    metrics = {"overall": regression_metrics(rows), "per_bucket": {}}
    for bucket_name in DISTANCE_BUCKETS:
        metrics["per_bucket"][bucket_name] = regression_metrics(by_bucket[bucket_name])
    return metrics


def plot_eval_figure(rows, metrics, output_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=180)
    ax_scatter, ax_hist, ax_resid, ax_bucket = axes.flatten()

    gt_all = [row["gt_speed_mpm"] for row in rows]
    pred_all = [row["pred_speed_mpm"] for row in rows]
    min_v = min(min(gt_all), min(pred_all))
    max_v = max(max(gt_all), max(pred_all))

    for bucket_name in DISTANCE_BUCKETS:
        bucket_rows = [row for row in rows if row["bucket_name"] == bucket_name]
        ax_scatter.scatter(
            [row["gt_speed_mpm"] for row in bucket_rows],
            [row["pred_speed_mpm"] for row in bucket_rows],
            s=18,
            alpha=0.65,
            label=f"{bucket_name} (n={len(bucket_rows)})",
            color=BUCKET_COLORS[bucket_name],
        )
    ax_scatter.plot([min_v, max_v], [min_v, max_v], linestyle="--", color="black", linewidth=1.2)
    ax_scatter.set_title(
        f"GT vs Pred ({EVAL_SPLIT})\n"
        f"MAE={metrics['overall']['mae_mpm']:.1f} m/min, "
        f"RMSE={metrics['overall']['rmse_mpm']:.1f}, "
        f"R2={metrics['overall']['r2']:.3f}"
    )
    ax_scatter.set_xlabel("GT Speed (m/min)")
    ax_scatter.set_ylabel("Pred Speed (m/min)")
    ax_scatter.grid(alpha=0.2, linestyle="--")
    ax_scatter.legend()

    ax_hist.hist([row["signed_error_mpm"] for row in rows], bins=60, color="#7F7F7F", edgecolor="white")
    ax_hist.axvline(0.0, color="#D62728", linestyle="--", linewidth=1.5)
    ax_hist.set_title("Signed Error Distribution")
    ax_hist.set_xlabel("Pred - GT (m/min)")
    ax_hist.set_ylabel("Count")
    ax_hist.grid(alpha=0.2, linestyle="--")

    for bucket_name in DISTANCE_BUCKETS:
        bucket_rows = [row for row in rows if row["bucket_name"] == bucket_name]
        ax_resid.scatter(
            [row["distance_km"] for row in bucket_rows],
            [row["signed_error_mpm"] for row in bucket_rows],
            s=18,
            alpha=0.6,
            color=BUCKET_COLORS[bucket_name],
            label=bucket_name,
        )
    ax_resid.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax_resid.set_title("Residual vs Distance")
    ax_resid.set_xlabel("Distance (km)")
    ax_resid.set_ylabel("Pred - GT (m/min)")
    ax_resid.grid(alpha=0.2, linestyle="--")

    bucket_names = DISTANCE_BUCKETS
    mae_vals = [metrics["per_bucket"][name]["mae_mpm"] for name in bucket_names]
    rmse_vals = [metrics["per_bucket"][name]["rmse_mpm"] for name in bucket_names]
    x = range(len(bucket_names))
    ax_bucket.bar([i - 0.18 for i in x], mae_vals, width=0.36, label="MAE", color="#4C78A8")
    ax_bucket.bar([i + 0.18 for i in x], rmse_vals, width=0.36, label="RMSE", color="#F58518")
    ax_bucket.set_xticks(list(x), bucket_names)
    ax_bucket.set_title("Error by Bucket")
    ax_bucket.set_ylabel("m/min")
    ax_bucket.grid(axis="y", alpha=0.2, linestyle="--")
    ax_bucket.legend()

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_by_threshold(metrics, output_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=180)
    bucket_names = DISTANCE_BUCKETS
    x = range(len(bucket_names))
    width = 0.22
    for offset_idx, thr in enumerate(ERROR_THRESHOLDS):
        values = [
            metrics["per_bucket"][name][f"acc_within_{int(thr)}mpm"] * 100.0
            for name in bucket_names
        ]
        ax.bar(
            [i + (offset_idx - 1) * width for i in x],
            values,
            width=width,
            label=f"<= {int(thr)} m/min",
        )
    ax.set_xticks(list(x), bucket_names)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Accuracy by Error Threshold ({EVAL_SPLIT})")
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_split_bucket_coverage(output_path: Path):
    split_bucket_counts = {}
    for split_name in ["val", "test"]:
        counts = Counter()
        with open(INDEX_DIR / f"{split_name}.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                counts[DISTANCE_BUCKETS[int(row["bucket_idx"])]] += 1
        split_bucket_counts[split_name] = counts

    fig, ax = plt.subplots(figsize=(8, 5), dpi=180)
    x = range(len(DISTANCE_BUCKETS))
    width = 0.35
    val_vals = [split_bucket_counts["val"][name] for name in DISTANCE_BUCKETS]
    test_vals = [split_bucket_counts["test"][name] for name in DISTANCE_BUCKETS]
    ax.bar([i - width / 2 for i in x], val_vals, width=width, label="val", color="#9C755F")
    ax.bar([i + width / 2 for i in x], test_vals, width=width, label="test", color="#BAB0AC")
    ax.set_xticks(list(x), DISTANCE_BUCKETS)
    ax.set_ylabel("Count")
    ax.set_title("Bucket Coverage in Val/Test")
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return {
        "val": {name: split_bucket_counts["val"][name] for name in DISTANCE_BUCKETS},
        "test": {name: split_bucket_counts["test"][name] for name in DISTANCE_BUCKETS},
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = evaluate_split(EVAL_SPLIT)
    metrics = build_metrics(rows)
    coverage = plot_split_bucket_coverage(OUTPUT_DIR / "val_test_bucket_coverage.png")

    plot_eval_figure(rows, metrics, OUTPUT_DIR / f"{EVAL_SPLIT}_model_eval.png")
    plot_accuracy_by_threshold(metrics, OUTPUT_DIR / f"{EVAL_SPLIT}_accuracy_by_threshold.png")
    save_predictions_csv(rows, OUTPUT_DIR / f"{EVAL_SPLIT}_predictions.csv")

    out = {
        "eval_split": EVAL_SPLIT,
        "checkpoint": str(CKPT_PATH),
        "metrics": metrics,
        "val_test_bucket_coverage": coverage,
    }
    with open(OUTPUT_DIR / f"{EVAL_SPLIT}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[eval] split={EVAL_SPLIT}, records={len(rows)}")
    print(f"[eval] metrics json: {OUTPUT_DIR / f'{EVAL_SPLIT}_metrics.json'}")
    print(f"[eval] eval fig: {OUTPUT_DIR / f'{EVAL_SPLIT}_model_eval.png'}")
    print(f"[eval] accuracy fig: {OUTPUT_DIR / f'{EVAL_SPLIT}_accuracy_by_threshold.png'}")
    print(f"[eval] coverage fig: {OUTPUT_DIR / 'val_test_bucket_coverage.png'}")
    print(f"[eval] predictions csv: {OUTPUT_DIR / f'{EVAL_SPLIT}_predictions.csv'}")


if __name__ == "__main__":
    main()
