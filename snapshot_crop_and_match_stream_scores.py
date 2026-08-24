#!/usr/bin/env python3
"""Snapshot current streamed-eye images, crop them, and join score records."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EYES_ROOT = REPO_ROOT / "data" / "E2E" / "eyes"
DEFAULT_IMAGE_DIR = DEFAULT_EYES_ROOT / "raw_video_streamed"
DEFAULT_RAW_VIDEO_ROOT = REPO_ROOT / "data" / "E2E" / "raw_video"
DEFAULT_SCORE_MANIFEST = (
    REPO_ROOT / "data" / "E2E" / "qinggewang_low_speed_800_all3.with_eye.full_manifest.jsonl"
)
DEFAULT_SCORE_SUMMARY = (
    REPO_ROOT
    / "data"
    / "E2E"
    / "qinggewang_low_speed_800_all3.with_eye.full_manifest.summary.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--raw-video-root", type=Path, default=DEFAULT_RAW_VIDEO_ROOT)
    parser.add_argument(
        "--webdav-base-url",
        default="https://webdav.123pan.cn/webdav",
        help="Used only to reconstruct remote_video_url in the snapshot manifest.",
    )
    parser.add_argument("--score-manifest", type=Path, default=DEFAULT_SCORE_MANIFEST)
    parser.add_argument("--score-summary-json", type=Path, default=DEFAULT_SCORE_SUMMARY)
    parser.add_argument("--eyes-root", type=Path, default=DEFAULT_EYES_ROOT)
    parser.add_argument("--size", type=int, default=518)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--tag", default=None, help="Output tag. Default is current local timestamp.")
    parser.add_argument("--output-subdir-prefix", default="raw_video_streamed_cc518")
    parser.add_argument("--output-prefix", default="raw_video.streamed.cc518")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every-crop", type=int, default=200)
    parser.add_argument("--progress-every-match", type=int, default=100000)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def dump_jsonl(rows: list[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def format_seconds(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "unknown"
    total_seconds = int(round(seconds))
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def print_crop_progress(
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
        f"Crop: {processed_count}/{total_count} ({percent:5.1f}%) | "
        f"written={written_count} skipped={skipped_count} failed={failed_count} | "
        f"{rate:.2f} items/s | ETA {format_seconds(eta_seconds)}"
    )
    if force_newline:
        print(line, flush=True)
    else:
        print("\r" + line, end="", flush=True)


def print_match_progress(
    scanned_count: int,
    total_count: int | None,
    matched_rows: int,
    matched_rings: int,
    started_at: float,
    force_newline: bool = False,
) -> None:
    elapsed = max(time.monotonic() - started_at, 1e-6)
    rate = scanned_count / elapsed
    if total_count:
        remaining = max(total_count - scanned_count, 0)
        eta_seconds = (remaining / rate) if rate > 0 else None
        percent = scanned_count / total_count * 100.0
        prefix = f"Match scan: {scanned_count}/{total_count} ({percent:5.1f}%)"
    else:
        eta_seconds = None
        prefix = f"Match scan: {scanned_count}"
    line = (
        f"{prefix} | matched_rows={matched_rows} matched_rings={matched_rings} | "
        f"{rate:.2f} rows/s | ETA {format_seconds(eta_seconds)}"
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


def parse_stream_image_name(image_path: Path) -> dict | None:
    stem = image_path.stem
    parts = stem.split("__")
    if len(parts) < 3 or not stem.endswith("_a"):
        return None
    collection = "__".join(parts[:-2])
    batch_id = parts[-2]
    ring_number = parts[-1][:-2]
    return {
        "collection": collection,
        "batch_id": batch_id,
        "ring_number": ring_number,
        "file_name": image_path.name,
    }


def infer_timestamp_and_source(metadata: dict) -> tuple[int | None, str | None]:
    selected_frame = metadata.get("selected_frame") or {}
    frame_decode = metadata.get("frame_decode") or {}
    selection = metadata.get("selection") or {}

    if selected_frame.get("timestamp_ms") is not None:
        return selected_frame.get("timestamp_ms"), "selected_frame.timestamp_ms"
    if frame_decode.get("decoded_timestamp_ms") is not None:
        return frame_decode.get("decoded_timestamp_ms"), "frame_decode.decoded_timestamp_ms"
    if selection.get("decoded_timestamp_ms") is not None:
        return selection.get("decoded_timestamp_ms"), "selection.decoded_timestamp_ms"
    return None, None


def build_remote_video_url(webdav_base_url: str | None, source_video_path: str | None) -> str | None:
    if not webdav_base_url or not source_video_path:
        return None
    base = webdav_base_url.rstrip("/")
    return base + "/" + quote(source_video_path, safe="/")


def build_metadata_index(raw_video_root: Path) -> dict[tuple[str, str, str], dict]:
    index: dict[tuple[str, str, str], dict] = {}
    for metadata_path in raw_video_root.rglob("metadata.json"):
        try:
            metadata = load_json(metadata_path)
        except Exception:
            continue
        source_video_path = metadata.get("source_video_path")
        batch_id = metadata.get("batch_id")
        if not source_video_path or not batch_id:
            continue
        source_path = Path(source_video_path)
        ring_number = source_path.stem
        collection = source_path.parent.parent.name if source_path.parent.name == "videos" else None
        if not collection:
            continue
        index[(collection, batch_id, ring_number)] = {
            "metadata": metadata,
            "metadata_abs_path": str(metadata_path.resolve()),
        }
    return index


def snapshot_stream_rows_from_images(
    image_dir: Path,
    raw_video_root: Path,
    snapshot_manifest: Path,
    webdav_base_url: str | None,
) -> tuple[list[dict], dict]:
    image_paths = sorted(image_dir.glob("*.jpg"))
    metadata_index = build_metadata_index(raw_video_root)
    rows: list[dict] = []
    failures: list[dict] = []
    metadata_hits = 0

    for image_path in image_paths:
        parsed = parse_stream_image_name(image_path)
        if parsed is None:
            failures.append(
                {
                    "image_file_name": image_path.name,
                    "error": "unsupported_image_name_format",
                }
            )
            continue

        key = (parsed["collection"], parsed["batch_id"], parsed["ring_number"])
        metadata_entry = metadata_index.get(key)
        metadata = metadata_entry["metadata"] if metadata_entry else {}
        if metadata_entry:
            metadata_hits += 1

        timestamp_ms, timestamp_source = infer_timestamp_and_source(metadata)
        selected_frame = metadata.get("selected_frame") or {}
        row = {
            "ring_number": parsed["ring_number"],
            "batch_id": parsed["batch_id"],
            "artifact_profile": metadata.get("artifact_profile"),
            "status": metadata.get("status", "succeeded"),
            "source_video_path": metadata.get("source_video_path"),
            "metadata_abs_path": metadata_entry["metadata_abs_path"] if metadata_entry else None,
            "image_abs_path": str(image_path.resolve()),
            "remote_video_url": build_remote_video_url(
                webdav_base_url, metadata.get("source_video_path")
            ),
            "timestamp_ms": timestamp_ms,
            "timestamp_source": timestamp_source,
            "bbox": selected_frame.get("bbox"),
            "crop_box_xyxy": None,
            "output_size": None,
            "snapshot_from": "image_dir",
        }
        rows.append(row)

    dump_jsonl(rows, snapshot_manifest)
    summary = {
        "image_dir": str(image_dir.resolve()),
        "raw_video_root": str(raw_video_root.resolve()),
        "snapshot_manifest": str(snapshot_manifest.resolve()),
        "image_files_found": len(image_paths),
        "snapshot_rows_written": len(rows),
        "metadata_index_size": len(metadata_index),
        "metadata_hits": metadata_hits,
        "metadata_misses": len(rows) - metadata_hits,
        "name_parse_failures": len(failures),
        "failure_examples": failures[:20],
    }
    return rows, summary


def crop_snapshot_rows(
    snapshot_rows: list[dict],
    eyes_root: Path,
    output_dir: Path,
    output_manifest: Path,
    size: int,
    jpeg_quality: int,
    overwrite: bool,
    progress_every: int,
) -> tuple[list[dict], dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_rows: list[dict] = []
    failures: list[dict] = []

    existing_outputs = 0
    missing_inputs = 0
    for row in snapshot_rows:
        src_path = Path(row["image_abs_path"])
        if not src_path.exists():
            missing_inputs += 1
            continue
        dst_path = output_image_path(output_dir, row)
        if dst_path.exists() and not overwrite:
            existing_outputs += 1

    plan_summary = {
        "rows_total": len(snapshot_rows),
        "output_dir": str(output_dir.resolve()),
        "target_size": size,
        "missing_input_images": missing_inputs,
        "already_exists_count": existing_outputs,
        "need_process_count": max(len(snapshot_rows) - missing_inputs - existing_outputs, 0),
    }
    print("Planned crop summary:")
    print(json.dumps(plan_summary, ensure_ascii=False, indent=2))

    processed_count = 0
    written_count = 0
    skipped_count = 0
    failed_count = 0
    started_at = time.monotonic()

    for row in snapshot_rows:
        src_path = Path(row["image_abs_path"])
        dst_path = output_image_path(output_dir, row)
        try:
            if not src_path.exists():
                raise FileNotFoundError(f"missing input image: {src_path}")

            if dst_path.exists() and not overwrite:
                out_row = dict(row)
                out_row.update(build_eye_fields(eyes_root, dst_path))
                out_row["preprocess"] = {
                    "type": "center_crop_resize",
                    "target_size": size,
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
                    resized = cropped.resize((size, size), Image.Resampling.BICUBIC)
                    resized.save(
                        dst_path,
                        format="JPEG",
                        quality=jpeg_quality,
                        optimize=True,
                    )

                out_row = dict(row)
                out_row.update(build_eye_fields(eyes_root, dst_path))
                out_row["preprocess"] = {
                    "type": "center_crop_resize",
                    "target_size": size,
                    "source_image_abs_path": str(src_path.resolve()),
                    "source_size": {"width": source_size[0], "height": source_size[1]},
                    "center_crop_xyxy": crop_box,
                }
                out_row["output_size"] = {"width": size, "height": size}
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
        if processed_count % max(progress_every, 1) == 0:
            print_crop_progress(
                processed_count=processed_count,
                total_count=len(snapshot_rows),
                written_count=written_count,
                skipped_count=skipped_count,
                failed_count=failed_count,
                started_at=started_at,
            )

    if processed_count % max(progress_every, 1) != 0:
        print_crop_progress(
            processed_count=processed_count,
            total_count=len(snapshot_rows),
            written_count=written_count,
            skipped_count=skipped_count,
            failed_count=failed_count,
            started_at=started_at,
            force_newline=True,
        )
    else:
        print()

    dump_jsonl(processed_rows, output_manifest)
    summary = {
        "rows_total": len(snapshot_rows),
        "output_manifest": str(output_manifest.resolve()),
        "output_dir": str(output_dir.resolve()),
        "plan_summary": plan_summary,
        "output_written": len(processed_rows),
        "written_count": written_count,
        "skipped_count": skipped_count,
        "failed_count": failed_count,
        "failure_examples": failures[:20],
    }
    return processed_rows, summary


def score_dedup_key(score_row: dict) -> tuple:
    return (
        score_row.get("ring_number"),
        score_row.get("record_index"),
        score_row.get("race_name"),
        score_row.get("release_time"),
        score_row.get("arrival_time"),
        score_row.get("distance_km"),
        score_row.get("speed_mpm"),
        score_row.get("rank_no"),
        score_row.get("match_detail_url"),
    )


def merge_score_with_stream(score_row: dict, stream_row: dict) -> dict:
    merged = dict(score_row)
    merged.update(
        {
            "image_abs_path": stream_row.get("image_abs_path"),
            "eye_closeup_count": stream_row.get("eye_closeup_count", 1),
            "eye_closeup_file_names": stream_row.get("eye_closeup_file_names", []),
            "eye_closeup_relative_paths": stream_row.get("eye_closeup_relative_paths", []),
            "eye_closeup_file_paths": stream_row.get("eye_closeup_file_paths", []),
            "eye_closeup_first_file_name": stream_row.get("eye_closeup_first_file_name"),
            "eye_closeup_first_relative_path": stream_row.get("eye_closeup_first_relative_path"),
            "eye_closeup_first_file_path": stream_row.get("eye_closeup_first_file_path"),
            "stream_batch_id": stream_row.get("batch_id"),
            "stream_status": stream_row.get("status"),
            "stream_source_video_path": stream_row.get("source_video_path"),
            "stream_metadata_abs_path": stream_row.get("metadata_abs_path"),
            "stream_remote_video_url": stream_row.get("remote_video_url"),
            "stream_timestamp_ms": stream_row.get("timestamp_ms"),
            "stream_timestamp_source": stream_row.get("timestamp_source"),
            "stream_bbox": stream_row.get("bbox"),
            "stream_crop_box_xyxy": stream_row.get("crop_box_xyxy"),
            "stream_output_size": stream_row.get("output_size"),
            "stream_preprocess": stream_row.get("preprocess"),
        }
    )
    return merged


def match_scores_for_stream(
    processed_rows: list[dict],
    score_manifest: Path,
    score_summary_json: Path | None,
    output_manifest: Path,
    progress_every: int,
) -> dict:
    ring_to_rows: dict[str, list[dict]] = defaultdict(list)
    for row in processed_rows:
        ring = row.get("ring_number")
        if ring:
            ring_to_rows[ring].append(row)

    target_rings = set(ring_to_rows)
    total_score_rows = None
    if score_summary_json and score_summary_json.exists():
        summary_obj = load_json(score_summary_json)
        total_score_rows = (
            summary_obj.get("rows_written")
            or summary_obj.get("total_rows_expected")
            or summary_obj.get("total_rows")
        )

    print("Planned match summary:")
    print(
        json.dumps(
            {
                "score_manifest": str(score_manifest.resolve()),
                "target_stream_rows": len(processed_rows),
                "target_rings": len(target_rings),
                "matched_output_manifest": str(output_manifest.resolve()),
                "score_rows_total": total_score_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    seen_per_ring: dict[str, set[tuple]] = defaultdict(set)
    matched_rows = 0
    matched_rings: set[str] = set()
    scanned_count = 0
    started_at = time.monotonic()

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with score_manifest.open("r", encoding="utf-8") as fin, output_manifest.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            if not line.strip():
                continue
            scanned_count += 1
            score_row = json.loads(line)
            ring = score_row.get("ring_number")
            if ring in target_rings:
                dedup_key = score_dedup_key(score_row)
                if dedup_key not in seen_per_ring[ring]:
                    seen_per_ring[ring].add(dedup_key)
                    matched_rings.add(ring)
                    for stream_row in ring_to_rows[ring]:
                        fout.write(
                            json.dumps(
                                merge_score_with_stream(score_row, stream_row),
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        matched_rows += 1

            if scanned_count % max(progress_every, 1) == 0:
                print_match_progress(
                    scanned_count=scanned_count,
                    total_count=total_score_rows,
                    matched_rows=matched_rows,
                    matched_rings=len(matched_rings),
                    started_at=started_at,
                )

    if scanned_count % max(progress_every, 1) != 0:
        print_match_progress(
            scanned_count=scanned_count,
            total_count=total_score_rows,
            matched_rows=matched_rows,
            matched_rings=len(matched_rings),
            started_at=started_at,
            force_newline=True,
        )
    else:
        print()

    unmatched_rings = sorted(target_rings - matched_rings)
    return {
        "score_manifest": str(score_manifest.resolve()),
        "score_summary_json": str(score_summary_json.resolve()) if score_summary_json else None,
        "matched_output_manifest": str(output_manifest.resolve()),
        "target_stream_rows": len(processed_rows),
        "target_rings": len(target_rings),
        "score_rows_scanned": scanned_count,
        "score_rows_total": total_score_rows,
        "matched_rows": matched_rows,
        "matched_rings": len(matched_rings),
        "unmatched_rings": len(unmatched_rings),
        "unmatched_ring_examples": unmatched_rings[:20],
    }


def main() -> None:
    args = parse_args()
    tag = args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    eyes_root = args.eyes_root.resolve()
    image_dir = args.image_dir.resolve()
    raw_video_root = args.raw_video_root.resolve()
    raw_video_output_root = raw_video_root

    snapshot_manifest = raw_video_output_root / f"raw_video.streamed.snapshot.{tag}.jsonl"
    cropped_manifest = raw_video_output_root / f"{args.output_prefix}.{tag}.jsonl"
    matched_manifest = raw_video_output_root / f"{args.output_prefix}.with_scores.{tag}.jsonl"
    summary_json = raw_video_output_root / f"{args.output_prefix}.with_scores.{tag}.summary.json"
    output_dir = eyes_root / f"{args.output_subdir_prefix}_{tag}"

    print("Output plan:")
    print(
        json.dumps(
            {
                "tag": tag,
                "image_dir": str(image_dir),
                "raw_video_root": str(raw_video_root),
                "snapshot_manifest": str(snapshot_manifest.resolve()),
                "cropped_manifest": str(cropped_manifest.resolve()),
                "matched_manifest": str(matched_manifest.resolve()),
                "summary_json": str(summary_json.resolve()),
                "output_dir": str(output_dir.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    snapshot_rows, snapshot_summary = snapshot_stream_rows_from_images(
        image_dir=image_dir,
        raw_video_root=raw_video_root,
        snapshot_manifest=snapshot_manifest.resolve(),
        webdav_base_url=args.webdav_base_url,
    )
    print(
        json.dumps(snapshot_summary, ensure_ascii=False, indent=2)
    )

    processed_rows, crop_summary = crop_snapshot_rows(
        snapshot_rows=snapshot_rows,
        eyes_root=eyes_root,
        output_dir=output_dir,
        output_manifest=cropped_manifest,
        size=args.size,
        jpeg_quality=args.jpeg_quality,
        overwrite=args.overwrite,
        progress_every=args.progress_every_crop,
    )

    match_summary = match_scores_for_stream(
        processed_rows=processed_rows,
        score_manifest=args.score_manifest.resolve(),
        score_summary_json=args.score_summary_json.resolve()
        if args.score_summary_json
        else None,
        output_manifest=matched_manifest,
        progress_every=args.progress_every_match,
    )

    summary = {
        "tag": tag,
        "image_dir": str(image_dir),
        "raw_video_root": str(raw_video_root),
        "snapshot_manifest": str(snapshot_manifest.resolve()),
        "cropped_manifest": str(cropped_manifest.resolve()),
        "matched_manifest": str(matched_manifest.resolve()),
        "summary_json": str(summary_json.resolve()),
        "output_dir": str(output_dir.resolve()),
        "snapshot_summary": snapshot_summary,
        "crop_summary": crop_summary,
        "match_summary": match_summary,
    }
    with summary_json.resolve().open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
