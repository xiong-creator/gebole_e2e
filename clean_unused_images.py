"""删除 data/E2E/eyes 中未被 JSONL 引用的图片。

默认只预览，不执行删除；加 --delete 才真正删除。
"""
import argparse
import json
from pathlib import Path

from build_index import IMG_EXTS, basename_variants, candidate_rel_paths, norm_rel_path
from config import IMAGE_ROOT, JSONL_PATH, JSONL_PATHS


def build_allowed_sets(jsonl_path: Path):
    allowed_rel = set()
    allowed_names = set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            for rel in candidate_rel_paths(row):
                allowed_rel.add(rel)
                allowed_names.add(Path(rel).name)

            first_name = row.get("eye_closeup_first_file_name")
            if isinstance(first_name, str) and first_name.strip():
                allowed_names.update(basename_variants(first_name))

            for name in row.get("eye_closeup_file_names") or []:
                if isinstance(name, str) and name.strip():
                    allowed_names.update(basename_variants(name))
    return allowed_rel, allowed_names


def get_jsonl_paths() -> list[Path]:
    paths = []
    seen = set()
    configured = JSONL_PATHS if "JSONL_PATHS" in globals() else [JSONL_PATH]
    for path in configured:
        p = Path(path)
        if p in seen:
            continue
        seen.add(p)
        paths.append(p)
    return paths


def find_unused_images(image_root: Path, allowed_rel: set[str], allowed_names: set[str]):
    unused = []
    kept = 0
    for path in image_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMG_EXTS:
            continue
        rel = norm_rel_path(str(path.relative_to(image_root)))
        if rel in allowed_rel or path.name in allowed_names:
            kept += 1
            continue
        unused.append(path)
    return unused, kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delete", action="store_true", help="真正删除未引用图片")
    ap.add_argument("--limit", type=int, default=100, help="最多预览多少条待删除路径")
    args = ap.parse_args()

    jsonl_paths = get_jsonl_paths()
    print("[clean] JSONLs:")
    for path in jsonl_paths:
        print(f"  - {path}")
    print(f"[clean] IMAGE_ROOT: {IMAGE_ROOT}")

    allowed_rel = set()
    allowed_names = set()
    for path in jsonl_paths:
        rels, names = build_allowed_sets(path)
        allowed_rel.update(rels)
        allowed_names.update(names)
    print(
        f"[clean] allowed relative paths={len(allowed_rel)} "
        f"allowed names={len(allowed_names)}"
    )

    unused, kept = find_unused_images(IMAGE_ROOT, allowed_rel, allowed_names)
    print(f"[clean] kept={kept} unused={len(unused)}")

    preview = unused[: max(0, args.limit)]
    for path in preview:
        print(f"[unused] {path}")
    if len(unused) > len(preview):
        print(f"[clean] ... 还有 {len(unused) - len(preview)} 条未展示")

    if not args.delete:
        print("[clean] dry-run 模式；确认无误后加 --delete 执行删除。")
        return

    for path in unused:
        path.unlink()
    print(f"[clean] 已删除 {len(unused)} 张未引用图片")


if __name__ == "__main__":
    main()
