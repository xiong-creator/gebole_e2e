#!/usr/bin/env python3
"""Extract eye crops from remote raw videos via WebDAV.

This script scans local `raw_video/**/metadata.json`, resolves the original
video path from `source_video_path`, fetches a single frame near the recorded
timestamp with `ffmpeg`, and then crops the eye region from the decoded frame.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Iterable
from urllib.parse import quote

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = REPO_ROOT / "data" / "E2E" / "raw_video"
DEFAULT_EYES_ROOT = REPO_ROOT / "data" / "E2E" / "eyes"
DEFAULT_OUTPUT_SUBDIR = "raw_video_streamed"
DEFAULT_OUTPUT_MANIFEST = (
    REPO_ROOT / "data" / "E2E" / "raw_video" / "raw_video.streamed.jsonl"
)
DEFAULT_SUMMARY_JSON = (
    REPO_ROOT / "data" / "E2E" / "raw_video" / "raw_video.streamed.summary.json"
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


def safe_token(value: str | None) -> str:
    if not value:
        return "unknown"
    keep: list[str] = []
    for ch in value:
        if ch.isalnum() or ch in {"-", "_"}:
            keep.append(ch)
        else:
            keep.append("_")
    token = "".join(keep).strip("_")
    return token or "unknown"


def scan_metadata_paths(raw_root: Path) -> list[Path]:
    return sorted(raw_root.rglob("metadata.json"))


def pick_bbox(metadata: dict) -> dict | None:
    candidates = [
        (metadata.get("selected_frame") or {}).get("bbox"),
        (metadata.get("final_inference") or {}).get("bbox"),
        (metadata.get("selection") or {}).get("bbox"),
    ]
    for bbox in candidates:
        if isinstance(bbox, dict):
            return bbox
    return None


def pick_timestamp_ms(metadata: dict) -> tuple[int | None, str | None]:
    selected_frame = metadata.get("selected_frame") or {}
    frame_decode = metadata.get("frame_decode") or {}
    final_inference = metadata.get("final_inference") or {}
    selection = metadata.get("selection") or {}
    candidates = [
        ("selected_frame.timestamp_ms", selected_frame.get("timestamp_ms")),
        ("frame_decode.decoded_timestamp_ms", frame_decode.get("decoded_timestamp_ms")),
        ("selection.decoded_timestamp_ms", selection.get("decoded_timestamp_ms")),
        ("final_inference.timestamp_ms", final_inference.get("timestamp_ms")),
        ("selection.timestamp_ms", selection.get("timestamp_ms")),
    ]
    for field_name, value in candidates:
        if value is None:
            continue
        try:
            return int(round(float(value))), field_name
        except (TypeError, ValueError):
            continue
    return None, None


def normalize_bbox(bbox: dict, width: int, height: int) -> tuple[int, int, int, int] | None:
    try:
        x = int(round(float(bbox["x"])))
        y = int(round(float(bbox["y"])))
        w = int(round(float(bbox["width"])))
        h = int(round(float(bbox["height"])))
    except (KeyError, TypeError, ValueError):
        return None

    if w <= 0 or h <= 0 or width <= 0 or height <= 0:
        return None

    left = max(0, min(x, width))
    top = max(0, min(y, height))
    right = max(left, min(x + w, width))
    bottom = max(top, min(y + h, height))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def auth_header(username: str, password: str) -> str:
    raw = f"{username}:{password}".encode("utf-8")
    return "Authorization: Basic " + base64.b64encode(raw).decode("ascii")


def build_remote_url(
    base_url: str,
    source_video_path: str,
    strip_prefix: str | None,
) -> str:
    normalized = source_video_path.lstrip("/")
    if strip_prefix:
        prefix = strip_prefix.strip("/")
        if prefix and normalized == prefix:
            normalized = ""
        elif prefix and normalized.startswith(prefix + "/"):
            normalized = normalized[len(prefix) + 1 :]
    encoded = "/".join(quote(part, safe="") for part in normalized.split("/") if part)
    return base_url.rstrip("/") + ("/" + encoded if encoded else "")


def extract_frame_png(
    ffmpeg_path: str,
    remote_url: str,
    timestamp_ms: int,
    authorization_header: str,
    timeout_sec: int,
) -> bytes:
    timestamp_sec = max(timestamp_ms, 0) / 1000.0
    command = [
        ffmpeg_path,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-headers",
        authorization_header + "\r\n",
        "-ss",
        f"{timestamp_sec:.3f}",
        "-i",
        remote_url,
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "pipe:1",
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=timeout_sec,
    )
    if result.returncode != 0 or not result.stdout:
        stderr_text = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(stderr_text or "ffmpeg failed to extract a frame")
    return result.stdout


def output_image_path(eyes_root: Path, output_subdir: str, metadata_path: Path, batch_id: str | None) -> Path:
    ring_number = safe_token(metadata_path.parent.name)
    batch_token = safe_token(batch_id)
    collection_token = safe_token(metadata_path.parent.parent.parent.name)
    out_dir = eyes_root / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{collection_token}__{batch_token}__{ring_number}_a.jpg"


def ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def maybe_resize(image: Image.Image, max_edge: int | None) -> Image.Image:
    if not max_edge or max_edge <= 0:
        return image
    resized = image.copy()
    resized.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    return resized


def build_row(
    metadata_path: Path,
    metadata: dict,
    image_path: Path,
    remote_url: str,
    timestamp_ms: int,
    timestamp_source: str,
    bbox: dict,
    crop_box: tuple[int, int, int, int],
    image_size: tuple[int, int],
) -> dict:
    return {
        "ring_number": metadata_path.parent.name,
        "batch_id": metadata.get("batch_id"),
        "artifact_profile": metadata.get("artifact_profile"),
        "status": metadata.get("status"),
        "source_video_path": metadata.get("source_video_path"),
        "metadata_abs_path": str(metadata_path.resolve()),
        "image_abs_path": str(image_path.resolve()),
        "remote_video_url": remote_url,
        "timestamp_ms": timestamp_ms,
        "timestamp_source": timestamp_source,
        "bbox": bbox,
        "crop_box_xyxy": list(crop_box),
        "output_size": {"width": image_size[0], "height": image_size[1]},
    }


def summarize_plan(
    metadata_paths: list[Path],
    eyes_root: Path,
    output_subdir: str,
    overwrite: bool,
) -> dict:
    status_counter: Counter[str] = Counter()
    skip_counter: Counter[str] = Counter()
    eligible_count = 0
    need_extract_count = 0
    already_exists_count = 0

    for metadata_path in metadata_paths:
        metadata = load_json(metadata_path)
        status_counter[str(metadata.get("status"))] += 1

        if metadata.get("status") != "succeeded":
            skip_counter["non_succeeded"] += 1
            continue

        source_video_path = metadata.get("source_video_path")
        if not source_video_path:
            skip_counter["missing_source_video_path"] += 1
            continue

        bbox = pick_bbox(metadata)
        if not bbox:
            skip_counter["missing_bbox"] += 1
            continue

        timestamp_ms, timestamp_source = pick_timestamp_ms(metadata)
        if timestamp_ms is None or timestamp_source is None:
            skip_counter["missing_timestamp_ms"] += 1
            continue

        eligible_count += 1
        output_path = output_image_path(
            eyes_root=eyes_root,
            output_subdir=output_subdir,
            metadata_path=metadata_path,
            batch_id=metadata.get("batch_id"),
        )
        if output_path.exists() and not overwrite:
            already_exists_count += 1
        else:
            need_extract_count += 1

    return {
        "metadata_total_scanned": len(metadata_paths),
        "eligible_extract_count": eligible_count,
        "need_extract_count": need_extract_count,
        "already_exists_count": already_exists_count,
        "status_counts": dict(sorted(status_counter.items())),
        "skip_counts_before_run": dict(sorted(skip_counter.items())),
    }


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
    extracted_count: int,
    existing_count: int,
    failed_count: int,
    skipped_count: int,
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
        f"new={extracted_count} existing={existing_count} failed={failed_count} "
        f"skipped={skipped_count} | {rate:.2f} items/s | ETA {format_seconds(eta_seconds)}"
    )
    if force_newline:
        print(line, flush=True)
    else:
        print("\r" + line, end="", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--eyes-root", type=Path, default=DEFAULT_EYES_ROOT)
    parser.add_argument("--output-subdir", default=DEFAULT_OUTPUT_SUBDIR)
    parser.add_argument("--output-manifest", type=Path, default=DEFAULT_OUTPUT_MANIFEST)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--webdav-base-url", required=True)
    parser.add_argument("--webdav-username", required=True)
    parser.add_argument("--webdav-password", required=True)
    parser.add_argument(
        "--source-path-strip-prefix",
        default=None,
        help="Strip this prefix from source_video_path before appending to WebDAV base URL.",
    )
    parser.add_argument("--max-edge", type=int, default=256)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise SystemExit("ffmpeg not found in PATH")

    raw_root = args.raw_root.resolve()
    eyes_root = args.eyes_root.resolve()
    authorization_header = auth_header(args.webdav_username, args.webdav_password)
    metadata_paths = scan_metadata_paths(raw_root)
    if args.limit is not None:
        metadata_paths = metadata_paths[: args.limit]

    plan_summary = summarize_plan(
        metadata_paths=metadata_paths,
        eyes_root=eyes_root,
        output_subdir=args.output_subdir,
        overwrite=args.overwrite,
    )
    print("Planned extraction summary:")
    print(json.dumps(plan_summary, ensure_ascii=False, indent=2))

    rows: list[dict] = []
    status_counter: Counter[str] = Counter()
    skip_counter: Counter[str] = Counter()
    failures: list[dict] = []
    processed_count = 0
    extracted_count = 0
    started_at = time.monotonic()
    progress_every = max(args.progress_every, 1)

    for metadata_path in metadata_paths:
        metadata = load_json(metadata_path)
        status_counter[str(metadata.get("status"))] += 1

        if metadata.get("status") != "succeeded":
            skip_counter["non_succeeded"] += 1
            processed_count += 1
            if processed_count % progress_every == 0:
                print_progress(
                    processed_count=processed_count,
                    total_count=len(metadata_paths),
                    extracted_count=extracted_count,
                    existing_count=skip_counter["already_exists"],
                    failed_count=skip_counter["extract_failed"],
                    skipped_count=(
                        skip_counter["non_succeeded"]
                        + skip_counter["missing_source_video_path"]
                        + skip_counter["missing_bbox"]
                        + skip_counter["missing_timestamp_ms"]
                    ),
                    started_at=started_at,
                )
            continue

        source_video_path = metadata.get("source_video_path")
        if not source_video_path:
            skip_counter["missing_source_video_path"] += 1
            processed_count += 1
            if processed_count % progress_every == 0:
                print_progress(
                    processed_count=processed_count,
                    total_count=len(metadata_paths),
                    extracted_count=extracted_count,
                    existing_count=skip_counter["already_exists"],
                    failed_count=skip_counter["extract_failed"],
                    skipped_count=(
                        skip_counter["non_succeeded"]
                        + skip_counter["missing_source_video_path"]
                        + skip_counter["missing_bbox"]
                        + skip_counter["missing_timestamp_ms"]
                    ),
                    started_at=started_at,
                )
            continue

        bbox = pick_bbox(metadata)
        if not bbox:
            skip_counter["missing_bbox"] += 1
            processed_count += 1
            if processed_count % progress_every == 0:
                print_progress(
                    processed_count=processed_count,
                    total_count=len(metadata_paths),
                    extracted_count=extracted_count,
                    existing_count=skip_counter["already_exists"],
                    failed_count=skip_counter["extract_failed"],
                    skipped_count=(
                        skip_counter["non_succeeded"]
                        + skip_counter["missing_source_video_path"]
                        + skip_counter["missing_bbox"]
                        + skip_counter["missing_timestamp_ms"]
                    ),
                    started_at=started_at,
                )
            continue

        timestamp_ms, timestamp_source = pick_timestamp_ms(metadata)
        if timestamp_ms is None or timestamp_source is None:
            skip_counter["missing_timestamp_ms"] += 1
            processed_count += 1
            if processed_count % progress_every == 0:
                print_progress(
                    processed_count=processed_count,
                    total_count=len(metadata_paths),
                    extracted_count=extracted_count,
                    existing_count=skip_counter["already_exists"],
                    failed_count=skip_counter["extract_failed"],
                    skipped_count=(
                        skip_counter["non_succeeded"]
                        + skip_counter["missing_source_video_path"]
                        + skip_counter["missing_bbox"]
                        + skip_counter["missing_timestamp_ms"]
                    ),
                    started_at=started_at,
                )
            continue

        output_path = output_image_path(
            eyes_root=eyes_root,
            output_subdir=args.output_subdir,
            metadata_path=metadata_path,
            batch_id=metadata.get("batch_id"),
        )
        if output_path.exists() and not args.overwrite:
            skip_counter["already_exists"] += 1
            rows.append(
                build_row(
                    metadata_path=metadata_path,
                    metadata=metadata,
                    image_path=output_path,
                    remote_url=build_remote_url(
                        args.webdav_base_url,
                        source_video_path,
                        args.source_path_strip_prefix,
                    ),
                    timestamp_ms=timestamp_ms,
                    timestamp_source=timestamp_source,
                    bbox=bbox,
                    crop_box=[0, 0, 0, 0],
                    image_size=(0, 0),
                )
            )
            processed_count += 1
            if processed_count % progress_every == 0:
                print_progress(
                    processed_count=processed_count,
                    total_count=len(metadata_paths),
                    extracted_count=extracted_count,
                    existing_count=skip_counter["already_exists"],
                    failed_count=skip_counter["extract_failed"],
                    skipped_count=(
                        skip_counter["non_succeeded"]
                        + skip_counter["missing_source_video_path"]
                        + skip_counter["missing_bbox"]
                        + skip_counter["missing_timestamp_ms"]
                    ),
                    started_at=started_at,
                )
            continue

        remote_url = build_remote_url(
            args.webdav_base_url,
            source_video_path,
            args.source_path_strip_prefix,
        )
        try:
            frame_bytes = extract_frame_png(
                ffmpeg_path=ffmpeg_path,
                remote_url=remote_url,
                timestamp_ms=timestamp_ms,
                authorization_header=authorization_header,
                timeout_sec=args.timeout_sec,
            )
            with Image.open(io.BytesIO(frame_bytes)) as frame:
                frame.load()
                frame = ensure_rgb(frame)
                crop_box = normalize_bbox(bbox, frame.width, frame.height)
                if crop_box is None:
                    raise RuntimeError("bbox is outside the decoded frame")
                crop = frame.crop(crop_box)
                crop = maybe_resize(crop, args.max_edge)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                crop.save(
                    output_path,
                    format="JPEG",
                    quality=args.jpeg_quality,
                    optimize=True,
                )
                rows.append(
                    build_row(
                        metadata_path=metadata_path,
                        metadata=metadata,
                        image_path=output_path,
                        remote_url=remote_url,
                        timestamp_ms=timestamp_ms,
                        timestamp_source=timestamp_source,
                        bbox=bbox,
                        crop_box=crop_box,
                        image_size=crop.size,
                    )
                )
                extracted_count += 1
        except Exception as exc:  # noqa: BLE001
            failures.append(
                {
                    "metadata_abs_path": str(metadata_path.resolve()),
                    "source_video_path": source_video_path,
                    "error": str(exc),
                }
            )
            skip_counter["extract_failed"] += 1
        processed_count += 1
        if processed_count % progress_every == 0:
            print_progress(
                processed_count=processed_count,
                total_count=len(metadata_paths),
                extracted_count=extracted_count,
                existing_count=skip_counter["already_exists"],
                failed_count=skip_counter["extract_failed"],
                skipped_count=(
                    skip_counter["non_succeeded"]
                    + skip_counter["missing_source_video_path"]
                    + skip_counter["missing_bbox"]
                    + skip_counter["missing_timestamp_ms"]
                ),
                started_at=started_at,
            )

    output_written = dump_jsonl(rows, args.output_manifest.resolve())
    if processed_count % progress_every != 0:
        print_progress(
            processed_count=processed_count,
            total_count=len(metadata_paths),
            extracted_count=extracted_count,
            existing_count=skip_counter["already_exists"],
            failed_count=skip_counter["extract_failed"],
            skipped_count=(
                skip_counter["non_succeeded"]
                + skip_counter["missing_source_video_path"]
                + skip_counter["missing_bbox"]
                + skip_counter["missing_timestamp_ms"]
            ),
            started_at=started_at,
            force_newline=True,
        )
    else:
        print()
    summary = {
        "raw_root": str(raw_root),
        "eyes_root": str(eyes_root),
        "output_image_dir": str((eyes_root / args.output_subdir).resolve()),
        "output_manifest": str(args.output_manifest.resolve()),
        "summary_json": str(args.summary_json.resolve()),
        "metadata_total_scanned": len(metadata_paths),
        "plan_summary": plan_summary,
        "output_written": output_written,
        "status_counts": dict(sorted(status_counter.items())),
        "skip_counts": dict(sorted(skip_counter.items())),
        "failure_count": len(failures),
        "failure_examples": failures[:20],
        "max_edge": args.max_edge,
    }
    args.summary_json.resolve().parent.mkdir(parents=True, exist_ok=True)
    with args.summary_json.resolve().open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
