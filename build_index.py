"""扫描 JSONL 数据清单 + 递归扫描 IMAGE_ROOT 下的图片，构建：
  1. 路径/文件名 → 绝对路径 的索引（图片可能在 IMAGE_ROOT 下任意深度的子目录中）
  2. 每个距离桶的合理速度范围 speed_range.json
  3. 训练/验证/测试划分（按 ring_number 分组，避免同一只鸽子出现在多个划分）
     train.jsonl / val.jsonl / test.jsonl

因为图片正在陆续上传，所以：
  - 只保留 IMAGE_ROOT 下已存在图片的 JSONL 记录
  - 只保留有 distance_km > 0 且 speed_mpm > 0 的记录（排除训放）
"""
import json
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

from config import (
    JSONL_PATH,
    IMAGE_ROOT,
    INDEX_DIR,
    DISTANCE_BUCKETS,
    DISTANCE_BOUNDARY_PATH,
    DISTANCE_BOUNDARIES,
    distance_to_bucket_idx_with_boundaries,
    SPEED_QUANTILE_LOW,
    SPEED_QUANTILE_HIGH,
    SPEED_RANGE_EXPAND_LOW,
    SPEED_RANGE_EXPAND_HIGH,
    SEED,
    VAL_RATIO,
    TEST_RATIO,
)


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def norm_rel_path(path_str: str) -> str:
    return path_str.replace("\\", "/").lstrip("./").strip("/")


def basename_variants(name: str) -> set[str]:
    variants = {name}
    stem = Path(name).stem
    suffix = Path(name).suffix
    if stem.endswith("_a") or stem.endswith("_b"):
        variants.add(f"{stem[:-2]}{suffix}")
    else:
        variants.add(f"{stem}_a{suffix}")
        variants.add(f"{stem}_b{suffix}")
    return variants


def scan_image_root(root: Path) -> tuple[dict, dict]:
    """递归扫描 root，返回:
    1. {basename/变体名: absolute_path}
    2. {相对路径: absolute_path}
    """
    if not root.exists():
        print(f"[scan_image_root] 图片目录不存在: {root}")
        return {}, {}
    by_name = {}
    by_rel = {}
    dup_name = 0
    dup_rel = 0
    for p in tqdm(root.rglob("*"), desc=f"scan {root}"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMG_EXTS:
            continue
        abs_path = str(p)
        rel = norm_rel_path(str(p.relative_to(root)))
        if rel in by_rel:
            dup_rel += 1
        else:
            by_rel[rel] = abs_path

        for name in basename_variants(p.name):
            if name in by_name:
                dup_name += 1
                continue
            by_name[name] = abs_path
    print(
        f"[scan_image_root] 找到 {len(by_rel)} 条相对路径索引, "
        f"{len(by_name)} 条文件名索引, 重复路径 {dup_rel}, 重复文件名 {dup_name}"
    )
    return by_name, by_rel


def candidate_rel_paths(row: dict) -> Iterable[str]:
    seen = set()
    for key in (
        "eye_closeup_first_file_path",
        "eye_closeup_first_relative_path",
    ):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            rel = norm_rel_path(value)
            # JSONL 中可能带有来源根目录，如“中信网/有成绩无性别/...”
            if rel.startswith("中信网/有成绩无性别/"):
                rel = rel.split("中信网/有成绩无性别/", 1)[1]
            if rel not in seen:
                seen.add(rel)
                yield rel

    for key in ("eye_closeup_file_paths", "eye_closeup_relative_paths"):
        values = row.get(key) or []
        if not isinstance(values, list):
            continue
        for value in values:
            if not isinstance(value, str) or not value.strip():
                continue
            rel = norm_rel_path(value)
            if rel.startswith("中信网/有成绩无性别/"):
                rel = rel.split("中信网/有成绩无性别/", 1)[1]
            if rel not in seen:
                seen.add(rel)
                yield rel


def resolve_image_path(row: dict, image_index: dict, rel_index: dict) -> str | None:
    for rel in candidate_rel_paths(row):
        abs_path = rel_index.get(rel)
        if abs_path is not None:
            return abs_path

    file_names = []
    first_name = row.get("eye_closeup_first_file_name")
    if isinstance(first_name, str) and first_name.strip():
        file_names.append(first_name)

    for name in row.get("eye_closeup_file_names") or []:
        if isinstance(name, str) and name.strip():
            file_names.append(name)

    seen = set()
    for name in file_names:
        if name in seen:
            continue
        seen.add(name)
        abs_path = image_index.get(name)
        if abs_path is not None:
            return abs_path
    return None


def iter_valid_records(jsonl_path: Path, image_index: dict, rel_index: dict):
    """迭代 JSONL 中同时满足 (有速度/距离/图片文件) 的记录，
    并把图片路径替换成本地绝对路径 (image_abs_path)
    """
    total = 0
    no_speed = 0
    no_dist = 0
    no_img = 0
    kept = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            speed = row.get("speed_mpm")
            distance = row.get("distance_km")
            if speed is None or speed <= 0:
                no_speed += 1
                continue
            if distance is None or distance <= 0:
                no_dist += 1
                continue
            abs_path = resolve_image_path(row, image_index, rel_index)
            if abs_path is None:
                no_img += 1
                continue

            record = {
                "ring_number": row.get("ring_number"),
                "image_abs_path": abs_path,
                "image_file_name": Path(abs_path).name,
                "distance_km": float(distance),
                "speed_mpm": float(speed),
                "race_name": row.get("race_name"),
            }
            kept += 1
            yield record

    print(
        f"[iter_valid_records] total={total} kept={kept} "
        f"skip_no_speed={no_speed} skip_no_dist={no_dist} skip_no_img={no_img}"
    )


def quantile_value(sorted_values, q: float) -> float:
    if not sorted_values:
        raise ValueError("sorted_values 不能为空")
    idx = min(len(sorted_values) - 1, int((len(sorted_values) - 1) * q))
    return float(sorted_values[idx])


def assign_bucket_indices(records, boundaries):
    assigned = []
    for r in records:
        rec = dict(r)
        rec["bucket_idx"] = distance_to_bucket_idx_with_boundaries(
            float(rec["distance_km"]), boundaries
        )
        assigned.append(rec)
    return assigned


def compute_speed_range(records):
    """按距离桶收集速度分布 → 计算分位数范围"""
    per_bucket = {i: [] for i in range(len(DISTANCE_BUCKETS))}
    for r in records:
        per_bucket[r["bucket_idx"]].append(r["speed_mpm"])

    speed_range = {}
    for idx, speeds in per_bucket.items():
        name = DISTANCE_BUCKETS[idx]
        if not speeds:
            print(f"[warn] bucket {name} 没有可用样本")
            speed_range[name] = {"low": 1.0, "high": 1.0, "count": 0}
            continue
        speeds_sorted = sorted(speeds)
        positive_speeds = [v for v in speeds_sorted if v > 0]
        if not positive_speeds:
            print(f"[warn] bucket {name} 没有正速度样本")
            speed_range[name] = {"low": 1.0, "high": 1.0, "count": len(speeds_sorted)}
            continue
        if len(speeds_sorted) < 10:
            print(f"[warn] bucket {name} 样本过少: {len(speeds_sorted)}")
        low_q = quantile_value(speeds_sorted, SPEED_QUANTILE_LOW)
        high_q = quantile_value(speeds_sorted, SPEED_QUANTILE_HIGH)
        observed_positive_min = positive_speeds[0]
        observed_max = speeds_sorted[-1]
        low = max(low_q - SPEED_RANGE_EXPAND_LOW, observed_positive_min)
        high = max(high_q + SPEED_RANGE_EXPAND_HIGH, observed_max)
        n = len(speeds_sorted)
        speed_range[name] = {
            "low": float(low),
            "high": float(high),
            "count": n,
            "median": float(speeds_sorted[n // 2]),
            "low_quantile": float(low_q),
            "high_quantile": float(high_q),
            "observed_positive_min": float(observed_positive_min),
            "observed_max": float(observed_max),
        }
    return speed_range


def split_by_ring(records, seed=SEED, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO):
    """按 ring_number 分组划分，避免数据泄露"""
    by_ring = {}
    for r in records:
        by_ring.setdefault(r["ring_number"], []).append(r)

    rings = list(by_ring.keys())
    rng = random.Random(seed)
    rng.shuffle(rings)

    n = len(rings)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    test_rings = set(rings[:n_test])
    val_rings = set(rings[n_test : n_test + n_val])

    splits = {"train": [], "val": [], "test": []}
    for ring, recs in by_ring.items():
        if ring in test_rings:
            splits["test"].extend(recs)
        elif ring in val_rings:
            splits["val"].extend(recs)
        else:
            splits["train"].extend(recs)

    print(
        f"[split_by_ring] rings train={len(rings) - len(test_rings) - len(val_rings)} "
        f"val={len(val_rings)} test={len(test_rings)}"
    )
    print(
        f"[split_by_ring] samples train={len(splits['train'])} "
        f"val={len(splits['val'])} test={len(splits['test'])}"
    )
    return splits


def dump_jsonl(records, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[dump_jsonl] wrote {len(records)} → {path}")


def print_samples(records, k: int = 5):
    print(f"[preview] 前 {min(k, len(records))} 条:")
    for r in records[:k]:
        print(
            f"  ring={r['ring_number']:<20} "
            f"dist={r['distance_km']:>6.1f}km  "
            f"speed={r['speed_mpm']:>8.2f} m/min  "
            f"bucket={DISTANCE_BUCKETS[r['bucket_idx']]:<6} "
            f"img={r['image_abs_path']}"
        )


def main():
    print(f"[build_index] JSONL: {JSONL_PATH}")
    print(f"[build_index] IMAGE_ROOT: {IMAGE_ROOT}")

    image_index, rel_index = scan_image_root(IMAGE_ROOT)
    if not image_index or not rel_index:
        print(
            "[error] 图片目录为空。请把鸽眼图片放到 IMAGE_ROOT 下（允许任意子目录），\n"
            "        然后重新执行 build_index.py。"
        )
        return

    records = list(iter_valid_records(JSONL_PATH, image_index, rel_index))
    print(f"[build_index] valid records: {len(records)}")
    if len(records) == 0:
        print("[error] 没有匹配到任何图片。请检查 IMAGE_ROOT 与 JSONL 的路径/文件名是否一致。")
        return

    splits = split_by_ring(records)
    boundaries = list(DISTANCE_BOUNDARIES)
    print(f"[build_index] distance boundaries (fixed): {boundaries}")
    with open(DISTANCE_BOUNDARY_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "bucket_names": DISTANCE_BUCKETS,
                "boundaries": boundaries,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    splits = {
        name: assign_bucket_indices(recs, boundaries) for name, recs in splits.items()
    }

    for name, recs in splits.items():
        bucket_cnt = Counter(DISTANCE_BUCKETS[r["bucket_idx"]] for r in recs)
        print(f"[build_index] {name} bucket 样本分布: {dict(bucket_cnt)}")

    print_samples(splits["train"], k=5)

    # 速度范围只用训练集统计
    speed_range = compute_speed_range(splits["train"])
    print("[build_index] speed_range (train only):")
    for k, v in speed_range.items():
        print(f"  {k}: {v}")

    with open(INDEX_DIR / "speed_range.json", "w", encoding="utf-8") as f:
        json.dump(speed_range, f, ensure_ascii=False, indent=2)

    for name, recs in splits.items():
        dump_jsonl(recs, INDEX_DIR / f"{name}.jsonl")

    print("[build_index] done")


if __name__ == "__main__":
    main()
