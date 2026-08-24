#!/usr/bin/env python3
"""Center-crop streamed eye images to square and resize to 518x518."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EYES_ROOT = REPO_ROOT / "data" / "E2E" / "eyes"
DEFAULT_INPUT_MANIFEST = (
    REPO_ROOT / "data" / "E2E" / "raw_video" / "raw_video.streamed.jsonl"
)
DEFAULT_OUTPUT_SUBDIR = "raw_video_streamed_cc518"
DEFAULT_OUTPUT_MANIFEST = (
    REPO_ROOT / "data" / "E2E" / "raw_video" / "raw_video.streamed.cc518.jsonl"
)
DEFAULT_SUMMARY_JSON = (
    REPO_ROOT / "data" / "E2E" / "raw_video" / "raw_video.streamed.cc518.summary.json"
)


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def dump_jsonl(rows: Iterable[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def format_seconds(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "unknown"
    total_seconds = int(round(seconds))
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def print_progress(
    processed_count: int,
    total_count: int,
    written_count: int,
    skipped_count: int,
    failed_count: int,
    started_at: float,
    force_newline: bool = False,
) -> None:
    elapsed = max(time.monotonic() - started_at, 1e-6)
    rate = processed_count / elapsed
    remaining = max(total_count - processed_count, 0)
    eta_seconds = (remaining / rate) if rate > 0 else None
    percent = (processed_count / total_count * 100.0) if total_count else 100.0
    line = (
        f"Progress: {processed_count}/{total_count} ({percent:5.1f}%) | "
        f"written={written_count} skipped={skipped_count} failed={failed_count} | "
        f"{rate:.2f} items/s | ETA {format_seconds(eta_seconds)}"
    )
    if force_newline:
        print(line, flush=True)
    else:
        print("\r" + line, end="", flush=True)


def output_image_path(output_dir: Path, record: dict) -> Path:
    src_name = Path(record["image_abs_path"]).stem
    return output_dir / f"{src_name}_cc518.jpg"


def center_crop_square(image: Image.Image) -> tuple[Image.Image, list[int]]:
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    right = left + side
    bottom = top + side
    cropped = image.crop((left, top, right, bottom))
    return cropped, [left, top, right, bottom]


def build_eye_fields(eyes_root: Path, output_path: Path) -> dict:
    rel_path = output_path.relative_to(eyes_root).as_posix()
    file_name = output_path.name
    return {
        "image_abs_path": str(output_path.resolve()),
        "eye_closeup_count": 1,
        "eye_closeup_file_names": [file_name],
        "eye_closeup_relative_paths": [rel_path],
        "eye_closeup_file_paths": [rel_path],
        "eye_closeup_first_file_name": file_name,
        "eye_closeup_first_relative_path": rel_path,
        "eye_closeup_first_file_path": rel_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", type=Path, default=DEFAULT_INPUT_MANIFEST)
    parser.add_argument("--eyes-root", type=Path, default=DEFAULT_EYES_ROOT)
    parser.add_argument("--output-subdir", default=DEFAULT_OUTPUT_SUBDIR)
    parser.add_argument("--output-manifest", type=Path, default=DEFAULT_OUTPUT_MANIFEST)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--size", type=int, default=518)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eyes_root = args.eyes_root.resolve()
    rows = load_jsonl(args.input_manifest.resolve())
    if args.limit is not None:
        rows = rows[: args.limit]

    output_dir = eyes_root / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_outputs = 0
    missing_inputs = 0
    for row in rows:
        src_path = Path(row["image_abs_path"])
        if not src_path.exists():
            missing_inputs += 1
            continue
        dst_path = output_image_path(output_dir, row)
        if dst_path.exists() and not args.overwrite:
            existing_outputs += 1

    plan_summary = {
        "input_manifest": str(args.input_manifest.resolve()),
        "rows_total": len(rows),
        "output_dir": str(output_dir.resolve()),
        "target_size": args.size,
        "missing_input_images": missing_inputs,
        "already_exists_count": existing_outputs,
        "need_process_count": max(len(rows) - missing_inputs - existing_outputs, 0),
    }
    print("Planned center-crop summary:")
    print(json.dumps(plan_summary, ensure_ascii=False, indent=2))

    processed_rows: list[dict] = []
    failures: list[dict] = []
    processed_count = 0
    written_count = 0
    skipped_count = 0
    failed_count = 0
    started_at = time.monotonic()
    progress_every = max(args.progress_every, 1)

    for row in rows:
        src_path = Path(row["image_abs_path"])
        dst_path = output_image_path(output_dir, row)
        try:
            if not src_path.exists():
                raise FileNotFoundError(f"missing input image: {src_path}")

            if dst_path.exists() and not args.overwrite:
                out_row = dict(row)
                out_row.update(build_eye_fields(eyes_root, dst_path))
                out_row["preprocess"] = {
                    "type": "center_crop_resize",
                    "target_size": args.size,
                    "source_image_abs_path": str(src_path.resolve()),
                    "output_already_exists": True,
                }
                processed_rows.append(out_row)
                skipped_count += 1
            else:
                with Image.open(src_path) as image:
                    image.load()
                    image = image.convert("RGB")
                    source_size = [image.width, image.height]
                    cropped, crop_box = center_crop_square(image)
                    resized = cropped.resize((args.size, args.size), Image.Resampling.BICUBIC)
                    resized.save(
                        dst_path,
                        format="JPEG",
                        quality=args.jpeg_quality,
                        optimize=True,
                    )

                out_row = dict(row)
                out_row.update(build_eye_fields(eyes_root, dst_path))
                out_row["preprocess"] = {
                    "type": "center_crop_resize",
                    "target_size": args.size,
                    "source_image_abs_path": str(src_path.resolve()),
                    "source_size": {"width": source_size[0], "height": source_size[1]},
                    "center_crop_xyxy": crop_box,
                }
                out_row["output_size"] = {"width": args.size, "height": args.size}
                processed_rows.append(out_row)
                written_count += 1
        except Exception as exc:  # noqa: BLE001
            failures.append(
                {
                    "ring_number": row.get("ring_number"),
                    "image_abs_path": row.get("image_abs_path"),
                    "error": str(exc),
                }
            )
            failed_count += 1

        processed_count += 1
        if processed_count % progress_every == 0:
            print_progress(
                processed_count=processed_count,
                total_count=len(rows),
                written_count=written_count,
                skipped_count=skipped_count,
                failed_count=failed_count,
                started_at=started_at,
            )

    if processed_count % progress_every != 0:
        print_progress(
            processed_count=processed_count,
            total_count=len(rows),
            written_count=written_count,
            skipped_count=skipped_count,
            failed_count=failed_count,
            started_at=started_at,
            force_newline=True,
        )
    else:
        print()

    output_written = dump_jsonl(processed_rows, args.output_manifest.resolve())
    summary = {
        "input_manifest": str(args.input_manifest.resolve()),
        "output_manifest": str(args.output_manifest.resolve()),
        "summary_json": str(args.summary_json.resolve()),
        "output_dir": str(output_dir.resolve()),
        "plan_summary": plan_summary,
        "rows_total": len(rows),
        "output_written": output_written,
        "written_count": written_count,
        "skipped_count": skipped_count,
        "failed_count": failed_count,
        "failure_examples": failures[:20],
        "target_size": args.size,
    }
    args.summary_json.resolve().parent.mkdir(parents=True, exist_ok=True)
    with args.summary_json.resolve().open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
