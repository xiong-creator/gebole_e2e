"""Visualize current E2E dataset distance/speed distributions."""
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

from config import DISTANCE_BOUNDARIES, DISTANCE_BUCKETS, INDEX_DIR


OUTPUT_DIR = Path(__file__).resolve().parent / "figures"
INDEX_FILES = ["train.jsonl", "val.jsonl", "test.jsonl"]
BUCKET_COLORS = {
    "short": "#4C78A8",
    "middle": "#F58518",
    "long": "#54A24B",
}
PANEL_BG = "#F8F9FB"
GRID = "#D9DDE7"
TEXT = "#1F2937"


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


def histogram(values, bins):
    lo = min(values)
    hi = max(values)
    if hi <= lo:
        hi = lo + 1.0
    width = (hi - lo) / bins
    counts = [0 for _ in range(bins)]
    for value in values:
        idx = int((value - lo) / width)
        if idx == bins:
            idx -= 1
        counts[idx] += 1
    edges = [lo + i * width for i in range(bins + 1)]
    return counts, edges


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


def svg_hist_panel(title, xlabel, values, color, bins, panel_x, panel_y, panel_w, panel_h,
                   extra_lines=None, legend_items=None):
    extra_lines = extra_lines or []
    legend_items = legend_items or []
    chart_margin_left = 58
    chart_margin_right = 20
    chart_margin_top = 36
    chart_margin_bottom = 42
    chart_x = panel_x + chart_margin_left
    chart_y = panel_y + chart_margin_top
    chart_w = panel_w - chart_margin_left - chart_margin_right
    chart_h = panel_h - chart_margin_top - chart_margin_bottom
    counts, edges = histogram(values, bins)
    max_count = max(counts) if counts else 1

    parts = [
        f'<rect x="{panel_x}" y="{panel_y}" width="{panel_w}" height="{panel_h}" fill="{PANEL_BG}" rx="12"/>',
        f'<text x="{panel_x + 18}" y="{panel_y + 24}" font-size="18" fill="{TEXT}" font-weight="bold">{title}</text>',
    ]
    for i in range(5):
        y = chart_y + chart_h * i / 4
        parts.append(
            f'<line x1="{chart_x}" y1="{y:.1f}" x2="{chart_x + chart_w}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>'
        )
        tick_value = int(round(max_count * (4 - i) / 4))
        parts.append(
            f'<text x="{chart_x - 8}" y="{y + 4:.1f}" font-size="11" fill="{TEXT}" text-anchor="end">{tick_value}</text>'
        )

    bar_w = chart_w / max(1, len(counts))
    for i, count in enumerate(counts):
        bar_h = 0 if max_count == 0 else chart_h * count / max_count
        x = chart_x + i * bar_w
        y = chart_y + chart_h - bar_h
        parts.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(bar_w - 1, 1):.2f}" height="{bar_h:.2f}" fill="{color}" opacity="0.85"/>'
        )

    value_min = edges[0]
    value_max = edges[-1]
    for line_value, line_color, line_label in extra_lines:
        ratio = 0.0 if value_max <= value_min else (line_value - value_min) / (value_max - value_min)
        x = chart_x + min(max(ratio, 0.0), 1.0) * chart_w
        parts.append(
            f'<line x1="{x:.2f}" y1="{chart_y}" x2="{x:.2f}" y2="{chart_y + chart_h}" '
            f'stroke="{line_color}" stroke-width="2" stroke-dasharray="6,4"/>'
        )
        parts.append(
            f'<text x="{x + 4:.2f}" y="{chart_y + 14}" font-size="11" fill="{line_color}">{line_label}</text>'
        )

    parts.extend(
        [
            f'<line x1="{chart_x}" y1="{chart_y + chart_h}" x2="{chart_x + chart_w}" y2="{chart_y + chart_h}" stroke="{TEXT}" stroke-width="1.2"/>',
            f'<line x1="{chart_x}" y1="{chart_y}" x2="{chart_x}" y2="{chart_y + chart_h}" stroke="{TEXT}" stroke-width="1.2"/>',
            f'<text x="{chart_x}" y="{chart_y + chart_h + 18}" font-size="11" fill="{TEXT}" text-anchor="start">{value_min:.1f}</text>',
            f'<text x="{chart_x + chart_w}" y="{chart_y + chart_h + 18}" font-size="11" fill="{TEXT}" text-anchor="end">{value_max:.1f}</text>',
            f'<text x="{chart_x + chart_w / 2}" y="{panel_y + panel_h - 8}" font-size="12" fill="{TEXT}" text-anchor="middle">{xlabel}</text>',
            f'<text x="{panel_x + 14}" y="{chart_y - 10}" font-size="12" fill="{TEXT}">Count</text>',
        ]
    )

    legend_x = panel_x + 18
    legend_y = panel_y + panel_h - 18
    for idx, (label, item_color) in enumerate(legend_items):
        dx = idx * 115
        parts.append(
            f'<rect x="{legend_x + dx}" y="{legend_y - 10}" width="12" height="12" fill="{item_color}" rx="2"/>'
        )
        parts.append(
            f'<text x="{legend_x + dx + 18}" y="{legend_y}" font-size="11" fill="{TEXT}">{label}</text>'
        )
    return "\n".join(parts)


def write_svg(path, width, height, content):
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        f'<rect width="{width}" height="{height}" fill="white"/>'
        f"{content}</svg>"
    )
    path.write_text(svg, encoding="utf-8")


def plot_distance_distribution(records):
    distances = [r["distance_km"] for r in records]
    content = svg_hist_panel(
        title="Distance Distribution",
        xlabel="Distance (km)",
        values=distances,
        color="#7F7F7F",
        bins=60,
        panel_x=20,
        panel_y=20,
        panel_w=1160,
        panel_h=520,
        extra_lines=[
            (DISTANCE_BOUNDARIES[0], "#D62728", f"b1={DISTANCE_BOUNDARIES[0]:.0f}"),
            (DISTANCE_BOUNDARIES[1], "#D62728", f"b2={DISTANCE_BOUNDARIES[1]:.0f}"),
        ],
        legend_items=[("all records", "#7F7F7F")],
    )
    out_path = OUTPUT_DIR / "distance_distribution.svg"
    write_svg(out_path, 1200, 560, content)
    return out_path


def plot_speed_distribution(records):
    speeds = [r["speed_mpm"] for r in records]
    by_bucket = defaultdict(list)
    for r in records:
        by_bucket[r["bucket_name"]].append(r["speed_mpm"])

    panel_specs = []
    x_positions = [20, 410, 800]
    for idx, name in enumerate(DISTANCE_BUCKETS):
        panel_specs.append(
            svg_hist_panel(
                title=f"{name} Speed Distribution",
                xlabel="Speed (m/min)",
                values=by_bucket[name],
                color=BUCKET_COLORS[name],
                bins=30,
                panel_x=x_positions[idx],
                panel_y=20,
                panel_w=380,
                panel_h=520,
                legend_items=[(f"{name} n={len(by_bucket[name])}", BUCKET_COLORS[name])],
            )
        )
    out_path = OUTPUT_DIR / "speed_distribution.svg"
    write_svg(out_path, 1200, 560, "\n".join(panel_specs))
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
