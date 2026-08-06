"""Visualize current E2E dataset distance/speed distributions."""
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import DISTANCE_BOUNDARIES, DISTANCE_BUCKETS, INDEX_DIR


OUTPUT_DIR = Path(__file__).resolve().parent / "figures"
INDEX_FILES = ["train.jsonl", "val.jsonl", "test.jsonl"]
BUCKET_COLORS = {
    "short": "#4C78A8",
    "middle": "#F58518",
    "long": "#54A24B",
}


def quantile(sorted_values, q):
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    idx = (len(sorted_values) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = idx - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def load_records():
    records = []
    split_counts = Counter()
    for name in INDEX_FILES:
        path = INDEX_DIR / name
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                row["split"] = path.stem
                row["bucket_name"] = DISTANCE_BUCKETS[int(row["bucket_idx"])]
                records.append(row)
                split_counts[path.stem] += 1
    return records, split_counts


def describe(values):
    values = sorted(float(v) for v in values)
    return {
        "count": len(values),
        "min": values[0],
        "p01": quantile(values, 0.01),
        "p10": quantile(values, 0.10),
        "median": median(values),
        "mean": mean(values),
        "p90": quantile(values, 0.90),
        "p99": quantile(values, 0.99),
        "max": values[-1],
    }


def build_summary(records, split_counts):
    distances = [r["distance_km"] for r in records]
    speeds = [r["speed_mpm"] for r in records]
    by_bucket = defaultdict(list)
    for r in records:
        by_bucket[r["bucket_name"]].append(r)

    summary = {
        "total_records": len(records),
        "split_counts": dict(split_counts),
        "distance_boundaries_km": DISTANCE_BOUNDARIES,
        "distance_km": describe(distances),
        "speed_mpm": describe(speeds),
        "bucket_counts": {name: len(by_bucket[name]) for name in DISTANCE_BUCKETS},
        "per_bucket": {},
    }
    for name in DISTANCE_BUCKETS:
        bucket_records = by_bucket[name]
        bucket_distances = [r["distance_km"] for r in bucket_records]
        bucket_speeds = [r["speed_mpm"] for r in bucket_records]
        summary["per_bucket"][name] = {
            "count": len(bucket_records),
            "distance_km": describe(bucket_distances),
            "speed_mpm": describe(bucket_speeds),
        }
    return summary


def add_bucket_lines(ax):
    for boundary in DISTANCE_BOUNDARIES:
        ax.axvline(boundary, color="#D62728", linestyle="--", linewidth=1.5, alpha=0.9)


def plot_distance_distribution(records):
    distances = [r["distance_km"] for r in records]
    distances_sorted = sorted(distances)
    distance_p99 = quantile(distances_sorted, 0.99)
    clipped = [v for v in distances if v <= distance_p99]
    short_focus = [v for v in distances if v <= 80]
    main_focus = [v for v in distances if v <= 600]
    middle_long_focus = [v for v in distances if 300 <= v <= 560]
    bucket_counts = Counter(r["bucket_name"] for r in records)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=180)
    ax_full, ax_main, ax_short, ax_bucket = axes.flatten()

    ax_full.hist(clipped, bins=80, color="#7F7F7F", edgecolor="white")
    add_bucket_lines(ax_full)
    ax_full.set_title(f"Main Range Distance (<= P99: {distance_p99:.1f} km)")
    ax_full.set_xlabel("Distance (km)")
    ax_full.set_ylabel("Count")
    ax_full.grid(alpha=0.2, linestyle="--")

    ax_main.hist(main_focus, bins=80, color="#4C78A8", edgecolor="white")
    add_bucket_lines(ax_main)
    ax_main.set_xlim(0, 600)
    ax_main.set_title("Zoomed Distance (0-600 km)")
    ax_main.set_xlabel("Distance (km)")
    ax_main.set_ylabel("Count")
    ax_main.grid(alpha=0.2, linestyle="--")

    ax_short.hist(short_focus, bins=40, color="#F58518", edgecolor="white")
    ax_short.set_xlim(0, 80)
    ax_short.set_title("Short-Race Focus (0-80 km)")
    ax_short.set_xlabel("Distance (km)")
    ax_short.set_ylabel("Count")
    ax_short.grid(alpha=0.2, linestyle="--")

    bucket_names = DISTANCE_BUCKETS
    bucket_values = [bucket_counts[name] for name in bucket_names]
    ax_bucket.bar(bucket_names, bucket_values, color=[BUCKET_COLORS[name] for name in bucket_names])
    ax_bucket.set_title("Bucket Counts")
    ax_bucket.set_xlabel("Bucket")
    ax_bucket.set_ylabel("Count")
    ax_bucket.grid(axis="y", alpha=0.2, linestyle="--")
    for idx, value in enumerate(bucket_values):
        ax_bucket.text(idx, value, str(value), ha="center", va="bottom", fontsize=10)

    inset = ax_main.inset_axes([0.55, 0.45, 0.4, 0.45])
    inset.hist(middle_long_focus, bins=60, color="#54A24B", edgecolor="white")
    inset.set_xlim(300, 560)
    inset.set_title("300-560 km", fontsize=10)
    inset.tick_params(labelsize=8)
    inset.grid(alpha=0.15, linestyle="--")

    fig.tight_layout()
    out_path = OUTPUT_DIR / "distance_distribution.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_speed_distribution(records):
    by_bucket = defaultdict(list)
    for r in records:
        by_bucket[r["bucket_name"]].append(r["speed_mpm"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=180, sharey=True)
    for idx, name in enumerate(DISTANCE_BUCKETS):
        axes[idx].hist(
            by_bucket[name],
            bins=60,
            color=BUCKET_COLORS[name],
            edgecolor="white",
            alpha=0.9,
        )
        axes[idx].set_title(f"{name} Speed (n={len(by_bucket[name])})")
        axes[idx].set_xlabel("Speed (m/min)")
        axes[idx].grid(alpha=0.2, linestyle="--")
    axes[0].set_ylabel("Count")

    fig.tight_layout()
    out_path = OUTPUT_DIR / "speed_distribution.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records, split_counts = load_records()
    summary = build_summary(records, split_counts)

    distance_path = plot_distance_distribution(records)
    speed_path = plot_speed_distribution(records)

    summary_path = OUTPUT_DIR / "distribution_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[viz] records={len(records)}")
    print(f"[viz] distance plot: {distance_path}")
    print(f"[viz] speed plot: {speed_path}")
    print(f"[viz] summary json: {summary_path}")


if __name__ == "__main__":
    main()
