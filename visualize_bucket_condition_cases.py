"""Visualize how one image's predicted speed changes under different bucket conditions."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import torch

from config import BACKBONE, DISTANCE_BUCKETS
from predict import load_model, predict_for_bucket, prepare_input, resolve_metadata_path, load_distance_boundaries
from dataset import load_speed_range


RUN_DIR = Path("/home/xiongyajiao/gebole/CausalFSFG_TMM/e2e/runs/run_20260806_024333")
CKPT_PATH = RUN_DIR / "best.pth"
TEST_PREDICTIONS_CSV = Path("/home/xiongyajiao/gebole/CausalFSFG_TMM/e2e/figures/model_eval/test_predictions.csv")
TEST_JSONL = Path("/home/xiongyajiao/gebole/CausalFSFG_TMM/e2e/index/test.jsonl")
OUTPUT_DIR = Path("/home/xiongyajiao/gebole/CausalFSFG_TMM/e2e/figures/model_eval")

BUCKET_COLORS = {
    "short": "#4C78A8",
    "middle": "#F58518",
    "long": "#54A24B",
}


def load_prediction_rows():
    rows = []
    with open(TEST_PREDICTIONS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["distance_km"] = float(row["distance_km"])
            row["bucket_idx"] = int(row["bucket_idx"])
            row["gt_speed_mpm"] = float(row["gt_speed_mpm"])
            row["pred_speed_mpm"] = float(row["pred_speed_mpm"])
            row["abs_error_mpm"] = float(row["abs_error_mpm"])
            row["signed_error_mpm"] = float(row["signed_error_mpm"])
            row["pred_pct"] = float(row["pred_pct"])
            rows.append(row)
    return rows


def load_test_records():
    mapping = {}
    with open(TEST_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            key = (row["ring_number"], row["image_file_name"])
            mapping[key] = row
    return mapping


def select_cases(rows):
    by_bucket = defaultdict(list)
    for row in rows:
        by_bucket[row["bucket_name"]].append(row)

    selected = []
    for bucket_name in DISTANCE_BUCKETS:
        bucket_rows = by_bucket[bucket_name]
        bucket_rows_sorted_by_gt = sorted(bucket_rows, key=lambda x: x["gt_speed_mpm"])
        median_row = bucket_rows_sorted_by_gt[len(bucket_rows_sorted_by_gt) // 2]
        hard_row = max(bucket_rows, key=lambda x: x["abs_error_mpm"])
        selected.append(("representative", median_row))
        if hard_row["ring_number"] != median_row["ring_number"] or hard_row["image_file_name"] != median_row["image_file_name"]:
            selected.append(("hard_case", hard_row))
    return selected


def run_all_bucket_predictions(image_path: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    speed_range_path = resolve_metadata_path(
        CKPT_PATH, "speed_range.json", RUN_DIR / "speed_range.json"
    )
    boundary_path = resolve_metadata_path(
        CKPT_PATH, "distance_boundaries.json", RUN_DIR / "distance_boundaries.json"
    )
    ranges = load_speed_range(speed_range_path)
    boundaries = load_distance_boundaries(boundary_path)
    model = load_model(CKPT_PATH, BACKBONE, device)
    image_tensor = prepare_input(image_path, device)

    preds = []
    for bucket_idx in range(len(DISTANCE_BUCKETS)):
        preds.append(predict_for_bucket(model, image_tensor, bucket_idx, ranges, boundaries))
    return preds


def visualize_cases(selected_cases, test_record_map):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(selected_cases), 2, figsize=(14, 3.6 * len(selected_cases)), dpi=180)
    if len(selected_cases) == 1:
        axes = [axes]

    exported = []
    model = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    speed_range_path = resolve_metadata_path(CKPT_PATH, "speed_range.json", RUN_DIR / "speed_range.json")
    boundary_path = resolve_metadata_path(CKPT_PATH, "distance_boundaries.json", RUN_DIR / "distance_boundaries.json")
    ranges = load_speed_range(speed_range_path)
    boundaries = load_distance_boundaries(boundary_path)
    model = load_model(CKPT_PATH, BACKBONE, device)

    for row_idx, (case_type, row) in enumerate(selected_cases):
        key = (row["ring_number"], row["image_file_name"])
        record = test_record_map[key]
        image_path = Path(record["image_abs_path"])
        image = Image.open(image_path).convert("RGB")
        image_tensor = prepare_input(image_path, device)
        preds = [predict_for_bucket(model, image_tensor, bucket_idx, ranges, boundaries) for bucket_idx in range(len(DISTANCE_BUCKETS))]

        ax_img, ax_bar = axes[row_idx]
        ax_img.imshow(image)
        ax_img.axis("off")
        ax_img.set_title(f"{row['bucket_name']} | {case_type}")

        pred_speeds = [item["predicted_speed_mpm"] for item in preds]
        bucket_names = [item["bucket"] for item in preds]
        colors = [BUCKET_COLORS[name] for name in bucket_names]
        bars = ax_bar.barh(bucket_names, pred_speeds, color=colors, alpha=0.9)
        ax_bar.axvline(row["gt_speed_mpm"], color="#D62728", linestyle="--", linewidth=1.5, label="GT speed")
        actual_bucket_name = row["bucket_name"]
        for bar, bucket_name in zip(bars, bucket_names):
            if bucket_name == actual_bucket_name:
                bar.set_edgecolor("black")
                bar.set_linewidth(2.0)
        for i, speed in enumerate(pred_speeds):
            ax_bar.text(speed, i, f" {speed:.1f}", va="center", fontsize=9)

        actual_pred = next(item for item in preds if item["bucket"] == actual_bucket_name)
        ax_bar.set_title(
            f"GT={row['gt_speed_mpm']:.1f} m/min | dist={row['distance_km']:.1f} km | "
            f"actual={actual_bucket_name} | actual-bucket err={actual_pred['predicted_speed_mpm'] - row['gt_speed_mpm']:+.1f}"
        )
        ax_bar.set_xlabel("Predicted Speed (m/min)")
        ax_bar.grid(axis="x", alpha=0.2, linestyle="--")
        ax_bar.legend(loc="lower right")

        detail_text = (
            f"type: {case_type}\n"
            f"ring: {row['ring_number']}\n"
            f"dist_km: {row['distance_km']:.1f}\n"
            f"gt_mpm: {row['gt_speed_mpm']:.1f}\n"
            f"image: {row['image_file_name']}"
        )
        ax_bar.text(
            1.02,
            0.98,
            detail_text,
            transform=ax_bar.transAxes,
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "#F7F7F7", "edgecolor": "#CCCCCC"},
        )

        exported.append(
            {
                "case_type": case_type,
                "ring_number": row["ring_number"],
                "image_file_name": row["image_file_name"],
                "image_abs_path": str(image_path),
                "race_name": row["race_name"],
                "distance_km": row["distance_km"],
                "gt_speed_mpm": row["gt_speed_mpm"],
                "actual_bucket": actual_bucket_name,
                "predictions": preds,
            }
        )

    fig.tight_layout()
    fig_path = OUTPUT_DIR / "test_bucket_condition_cases.png"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

    json_path = OUTPUT_DIR / "test_bucket_condition_cases.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(exported, f, ensure_ascii=False, indent=2)

    print(f"[cases] fig: {fig_path}")
    print(f"[cases] json: {json_path}")


def main():
    rows = load_prediction_rows()
    test_record_map = load_test_records()
    selected_cases = select_cases(rows)
    visualize_cases(selected_cases, test_record_map)


if __name__ == "__main__":
    main()
