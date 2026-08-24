#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE_PATH="/home/xiongyajiao/gebole/CausalFSFG_TMM/data/E2E/eyes/20251118/2025年云南宣威银集春棚决赛获奖鸽照片欣赏/2024-01-0225017_a.jpg"
CKPT_PATH="$SCRIPT_DIR/checkpoints/best.pth"
python "$SCRIPT_DIR/predict.py" --image "$IMAGE_PATH" --ckpt "$CKPT_PATH" --all-buckets
