#!/usr/bin/env python3
"""Prepare raw_video eye crops into a training-friendly manifest.

By default this script only exports samples that already have a directly
available eye image on disk, and writes:
1. one unified image directory under data/E2E/eyes/
2. one unified JSONL manifest for training

RGBA / LA / transparent images are flattened onto a green background and
saved as RGB JPEG so the downstream dataset is visually consistent.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = REPO_ROOT / "data" / "E2E" / "raw_video"
DEFAULT_EYES_ROOT = REPO_ROOT / "data" / "E2E" / "eyes"
DEFAULT_OUTPUT_SUBDIR = "raw_video_trainable"
DEFAULT_OUTPUT_MANIFEST = (
    REPO_ROOT / "data" / "E2E" / "raw_video" / "raw_video.trainable.jsonl"
)
DEFAULT_METADATA_ONLY_MANIFEST = (
    REPO_ROOT
    / "data"
    / "E2E"
    / "raw_video"
    / "raw_video.metadata_only.manifest.jsonl"
)
DEFAULT_SUMMARY_JSON = (
    REPO_ROOT / "data" / "E2E" / "raw_video" / "raw_video.prepare_summary.json"
)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_jsonl(rows: Iterable[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def rel_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def safe_collection_parts(source_video_path: str | None, metadata_path: Path) -> tuple[str, str]:
    if source_video_path:
        parts = Path(source_video_path).parts
        if len(parts) >= 2:
            return parts[0], parts[1]
        if len(parts) == 1:
            return "unknown", parts[0]

    md_parts = metadata_path.parts
    try:
        idx = md_parts.index("raw_video")
        if len(md_parts) > idx + 3:
            source_root = md_parts[idx + 1]
            collection = md_parts[idx + 2]
            return source_root, collection
    except ValueError:
        pass
    return "unknown", metadata_path.parent.parent.name


def first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "existing"
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def export_rgb_image(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "existing"

    with Image.open(src) as img:
        img.load()
        if img.mode in {"RGBA", "LA"} or (
            img.mode == "P" and "transparency" in img.info
        ):
            rgba = img.convert("RGBA")
            bg = Image.new("RGBA", rgba.size, (0, 255, 0, 255))
            rgb = Image.alpha_composite(bg, rgba).convert("RGB")
            convert_mode = "rgba_to_rgb_green"
        else:
            rgb = img.convert("RGB")
            convert_mode = "rgb_copy"
        rgb.save(dst, format="JPEG", quality=95)
    return convert_mode


def safe_file_token(value: str | None) -> str:
    if not value:
        return "unknown"
    keep = []
    for ch in value:
        if ch.isalnum() or ch in {"-", "_"}:
            keep.append(ch)
        else:
            keep.append("_")
    token = "".join(keep).strip("_")
    return token or "unknown"


def build_eye_fields(eyes_root: Path, image_path: Path) -> dict:
    rel_path = rel_posix(image_path, eyes_root)
    file_name = image_path.name
    return {
        "image_abs_path": str(image_path.resolve()),
        "eye_closeup_count": 1,
        "eye_closeup_file_names": [file_name],
        "eye_closeup_relative_paths": [rel_path],
        "eye_closeup_file_paths": [rel_path],
        "eye_closeup_first_file_name": file_name,
        "eye_closeup_first_relative_path": rel_path,
        "eye_closeup_first_file_path": rel_path,
    }


def empty_eye_fields() -> dict:
    return {
        "image_abs_path": None,
        "eye_closeup_count": 0,
        "eye_closeup_file_names": [],
        "eye_closeup_relative_paths": [],
        "eye_closeup_file_paths": [],
        "eye_closeup_first_file_name": None,
        "eye_closeup_first_relative_path": None,
        "eye_closeup_first_file_path": None,
    }


def scan_metadata_paths(raw_root: Path) -> list[Path]:
    return sorted(raw_root.rglob("metadata.json"))


def build_base_record(metadata_path: Path, metadata: dict) -> dict:
    ring_number = metadata_path.parent.name
    source_video_path = metadata.get("source_video_path")
    source_root, source_collection = safe_collection_parts(source_video_path, metadata_path)
    artifact_profile = metadata.get("artifact_profile")
    if not artifact_profile and first_existing(
        [
            metadata_path.with_name("best_eye.png"),
            metadata_path.with_name("best_eye.jpg"),
            metadata_path.with_name("best_eye.jpeg"),
            metadata_path.with_name("best_eye.webp"),
        ]
    ):
        artifact_profile = "legacy-images-v1"

    selected_frame = metadata.get("selected_frame") or {}
    frame_decode = metadata.get("frame_decode") or {}
    final_inference = metadata.get("final_inference") or {}
    selection = metadata.get("selection") or {}

    record = {
        "ring_number": ring_number,
        "artifact_profile": artifact_profile,
        "schema_version": metadata.get("schema_version"),
        "status": metadata.get("status"),
        "batch_id": metadata.get("batch_id"),
        "source_root": source_root,
        "source_collection": source_collection,
        "source_video_path": source_video_path,
        "metadata_abs_path": str(metadata_path.resolve()),
        "selected_frame": selected_frame or None,
        "frame_decode": frame_decode or None,
        "selection": selection or None,
        "final_inference": final_inference or None,
        "selected_timestamp_ms": selected_frame.get("timestamp_ms")
        or final_inference.get("timestamp_ms")
        or selection.get("timestamp_ms"),
        "selected_frame_index": selected_frame.get("frame_index")
        or frame_decode.get("frame_index")
        or final_inference.get("frame_index"),
        "selected_bbox": selected_frame.get("bbox")
        or final_inference.get("bbox")
        or selection.get("bbox"),
    }
    return record


def output_image_path(
    eyes_root: Path, output_subdir: str, record: dict, source_image_path: Path
) -> Path:
    out_dir = eyes_root / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    source_root = safe_file_token(str(record.get("source_root")))
    batch_id = safe_file_token(str(record.get("batch_id")))
    ring_number = safe_file_token(str(record.get("ring_number")))
    return out_dir / f"{source_root}__{batch_id}__{ring_number}_a.jpg"


def prepare_records(
    raw_root: Path, eyes_root: Path, output_subdir: str
) -> tuple[list[dict], list[dict], dict]:
    with_image_rows: list[dict] = []
    metadata_only_rows: list[dict] = []
    image_export_mode_counter: Counter[str] = Counter()
    artifact_counter: Counter[str] = Counter()
    status_counter: Counter[str] = Counter()
    broken_metadata: list[str] = []

    metadata_paths = scan_metadata_paths(raw_root)
    for metadata_path in metadata_paths:
        try:
            metadata = load_json(metadata_path)
        except (OSError, json.JSONDecodeError):
            broken_metadata.append(str(metadata_path))
            continue

        record = build_base_record(metadata_path, metadata)
        artifact_counter[str(record.get("artifact_profile"))] += 1
        status_counter[str(record.get("status"))] += 1

        best_eye_src = first_existing(
            [
                metadata_path.with_name("best_eye.png"),
                metadata_path.with_name("best_eye.jpg"),
                metadata_path.with_name("best_eye.jpeg"),
                metadata_path.with_name("best_eye.webp"),
            ]
        )
        best_frame_src = first_existing(
            [
                metadata_path.with_name("best_frame.jpg"),
                metadata_path.with_name("best_frame.jpeg"),
                metadata_path.with_name("best_frame.png"),
                metadata_path.with_name("best_frame.webp"),
            ]
        )

        record["best_frame_abs_path"] = (
            str(best_frame_src.resolve()) if best_frame_src is not None else None
        )

        if best_eye_src is not None:
            dst_path = output_image_path(eyes_root, output_subdir, record, best_eye_src)
            image_export_mode_counter[export_rgb_image(best_eye_src, dst_path)] += 1
            record["best_eye_abs_path"] = str(dst_path.resolve())
            record.update(build_eye_fields(eyes_root, dst_path))
            with_image_rows.append(record)
        else:
            record["best_eye_abs_path"] = None
            record.update(empty_eye_fields())
            metadata_only_rows.append(record)

    summary = {
        "raw_root": str(raw_root),
        "eyes_root": str(eyes_root),
        "metadata_total": len(metadata_paths),
        "with_image_count": len(with_image_rows),
        "metadata_only_count": len(metadata_only_rows),
        "broken_metadata_count": len(broken_metadata),
        "broken_metadata_examples": broken_metadata[:20],
        "artifact_profile_counts": dict(sorted(artifact_counter.items())),
        "status_counts": dict(sorted(status_counter.items())),
        "image_export_modes": dict(sorted(image_export_mode_counter.items())),
        "output_image_dir": str((eyes_root / output_subdir).resolve()),
    }
    return with_image_rows, metadata_only_rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--eyes-root", type=Path, default=DEFAULT_EYES_ROOT)
    parser.add_argument(
        "--output-subdir",
        default=DEFAULT_OUTPUT_SUBDIR,
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=DEFAULT_OUTPUT_MANIFEST,
    )
    parser.add_argument(
        "--metadata-only-manifest",
        type=Path,
        default=None,
    )
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root.resolve()
    eyes_root = args.eyes_root.resolve()

    with_image_rows, metadata_only_rows, summary = prepare_records(
        raw_root, eyes_root, args.output_subdir
    )
    output_count = dump_jsonl(with_image_rows, args.output_manifest.resolve())
    summary["output_manifest"] = str(args.output_manifest.resolve())
    summary["output_written"] = output_count

    if args.metadata_only_manifest is not None:
        metadata_only_count = dump_jsonl(
            metadata_only_rows, args.metadata_only_manifest.resolve()
        )
        summary["metadata_only_manifest"] = str(args.metadata_only_manifest.resolve())
        summary["metadata_only_written"] = metadata_only_count

    args.summary_json.resolve().parent.mkdir(parents=True, exist_ok=True)
    with args.summary_json.resolve().open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
