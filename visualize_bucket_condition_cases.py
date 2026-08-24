"""Visualize how one image's predicted speed changes under different bucket conditions."""
import argparse
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


RUN_DIR = Path("/home/xiongyajiao/gebole/CausalFSFG_TMM/e2e/runs/run_20260824_074343")
CKPT_PATH = RUN_DIR / "best.pth"
TEST_PREDICTIONS_CSV = Path("/home/xiongyajiao/gebole/CausalFSFG_TMM/e2e/figures/model_eval_new/test_predictions.csv")
TEST_JSONL = Path("/home/xiongyajiao/gebole/CausalFSFG_TMM/e2e/index/test.jsonl")
OUTPUT_DIR = Path("/home/xiongyajiao/gebole/CausalFSFG_TMM/e2e/figures/model_eval_new")

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--ckpt-path", type=Path, default=None)
    parser.add_argument("--test-predictions-csv", type=Path, default=TEST_PREDICTIONS_CSV)
    parser.add_argument("--test-jsonl", type=Path, default=TEST_JSONL)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def load_prediction_rows(test_predictions_csv: Path):
    rows = []
    with open(test_predictions_csv, "r", encoding="utf-8") as f:
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


def load_test_records(test_jsonl: Path):
    mapping = {}
    with open(test_jsonl, "r", encoding="utf-8") as f:
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


def run_all_bucket_predictions(image_path: Path, ckpt_path: Path, run_dir: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    speed_range_path = resolve_metadata_path(
        ckpt_path, "speed_range.json", run_dir / "speed_range.json"
    )
    boundary_path = resolve_metadata_path(
        ckpt_path, "distance_boundaries.json", run_dir / "distance_boundaries.json"
    )
    ranges = load_speed_range(speed_range_path)
    boundaries = load_distance_boundaries(boundary_path)
    model = load_model(ckpt_path, BACKBONE, device)
    image_tensor = prepare_input(image_path, device)

    preds = []
    for bucket_idx in range(len(DISTANCE_BUCKETS)):
        preds.append(predict_for_bucket(model, image_tensor, bucket_idx, ranges, boundaries))
    return preds


def visualize_cases(selected_cases, test_record_map, ckpt_path: Path, run_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(selected_cases), 2, figsize=(14, 3.6 * len(selected_cases)), dpi=180)
    if len(selected_cases) == 1:
        axes = [axes]

    exported = []
    model = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    speed_range_path = resolve_metadata_path(ckpt_path, "speed_range.json", run_dir / "speed_range.json")
    boundary_path = resolve_metadata_path(ckpt_path, "distance_boundaries.json", run_dir / "distance_boundaries.json")
    ranges = load_speed_range(speed_range_path)
    boundaries = load_distance_boundaries(boundary_path)
    model = load_model(ckpt_path, BACKBONE, device)

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
    fig_path = output_dir / "test_bucket_condition_cases.png"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

    json_path = output_dir / "test_bucket_condition_cases.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(exported, f, ensure_ascii=False, indent=2)

    print(f"[cases] fig: {fig_path}")
    print(f"[cases] json: {json_path}")


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    ckpt_path = args.ckpt_path.resolve() if args.ckpt_path else (run_dir / "best.pth")
    rows = load_prediction_rows(args.test_predictions_csv.resolve())
    test_record_map = load_test_records(args.test_jsonl.resolve())
    selected_cases = select_cases(rows)
    visualize_cases(
        selected_cases,
        test_record_map,
        ckpt_path=ckpt_path,
        run_dir=run_dir,
        output_dir=args.output_dir.resolve(),
    )


if __name__ == "__main__":
    main()
