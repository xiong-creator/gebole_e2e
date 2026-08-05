"""训练脚本：冻结 DINOv2 提取 tokens，只训练 Attention Probe。
损失 = MSE(pct_pred, pct_target)。
最终速度 = low + pct * (high - low)，天然被限制在合理范围内。
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    DISTANCE_BOUNDARY_PATH,
    INDEX_DIR,
    OUT_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
    EPOCHS,
    LR,
    WEIGHT_DECAY,
    BACKBONE,
)
from dataset import PigeonEyeSpeedDataset, load_speed_range
from model import EyeSpeedNet


def build_loaders():
    ranges = load_speed_range()
    train_ds = PigeonEyeSpeedDataset(INDEX_DIR / "train.jsonl", ranges, train=True)
    val_ds = PigeonEyeSpeedDataset(INDEX_DIR / "val.jsonl", ranges, train=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )
    return train_loader, val_loader, ranges


def decode_speed(pct, low, high):
    return low + pct * (high - low)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_mse_pct = 0.0
    total_mae_speed = 0.0
    n = 0
    for batch in tqdm(loader, desc="val", leave=False):
        image = batch["image"].to(device, non_blocking=True)
        bucket = batch["bucket_idx"].to(device, non_blocking=True)
        target = batch["target_pct"].to(device, non_blocking=True)
        low = batch["low"].to(device, non_blocking=True)
        high = batch["high"].to(device, non_blocking=True)
        gt_speed = batch["speed_mpm"].to(device, non_blocking=True)

        pct = model(image, bucket)
        pred_speed = decode_speed(pct, low, high)

        bs = image.size(0)
        total_mse_pct += ((pct - target) ** 2).sum().item()
        total_mae_speed += (pred_speed - gt_speed).abs().sum().item()
        n += bs

    return {
        "mse_pct": total_mse_pct / max(1, n),
        "mae_speed_mpm": total_mae_speed / max(1, n),
    }


def train_one_epoch(model, loader, optim, loss_fn, device, epoch, scaler, amp):
    model.train()
    total_loss = 0.0
    n = 0
    pbar = tqdm(loader, desc=f"epoch {epoch} train")
    for batch in pbar:
        image = batch["image"].to(device, non_blocking=True)
        bucket = batch["bucket_idx"].to(device, non_blocking=True)
        target = batch["target_pct"].to(device, non_blocking=True)

        optim.zero_grad()
        if amp:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                pct = model(image, bucket)
                loss = loss_fn(pct, target)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            pct = model(image, bucket)
            loss = loss_fn(pct, target)
            loss.backward()
            optim.step()

        bs = image.size(0)
        total_loss += loss.item() * bs
        n += bs
        pbar.set_postfix(loss=f"{total_loss / n:.4f}")

    return total_loss / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--backbone", type=str, default=BACKBONE)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.set_defaults(amp=True)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--tag", type=str, default=time.strftime("%Y%m%d_%H%M%S"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = args.amp and device.type == "cuda"
    print(f"[train] device={device}, backbone={args.backbone}, amp={amp}")

    train_loader, val_loader, ranges = build_loaders()
    print(f"[train] speed ranges: {ranges}")
    print(f"[train] train batches={len(train_loader)}, val batches={len(val_loader)}")

    model = EyeSpeedNet(backbone=args.backbone).to(device)
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[train] params total={n_total / 1e6:.2f}M "
        f"trainable={n_trainable / 1e6:.2f}M (probe only, DINOv2 frozen)"
    )

    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"[train] resumed from {args.resume}")

    # 只把可训练参数（Attention Probe）交给 optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=WEIGHT_DECAY
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    loss_fn = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    run_dir = Path(OUT_DIR) / f"run_{args.tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train] output dir: {run_dir}")

    # 将当前索引对应的分桶边界和速度上下限固化到本次训练目录，避免后续重建索引后推理漂移。
    shutil.copy2(INDEX_DIR / "speed_range.json", run_dir / "speed_range.json")
    if DISTANCE_BOUNDARY_PATH.exists():
        shutil.copy2(DISTANCE_BOUNDARY_PATH, run_dir / "distance_boundaries.json")

    best_metric = float("inf")
    log_path = run_dir / "log.jsonl"
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optim, loss_fn, device, epoch, scaler, amp
        )
        val_metrics = evaluate(model, val_loader, device)
        sched.step()

        rec = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_mse_pct": val_metrics["mse_pct"],
            "val_mae_speed_mpm": val_metrics["mae_speed_mpm"],
            "lr": optim.param_groups[0]["lr"],
            "time_sec": time.time() - t0,
        }
        print(f"[epoch {epoch}] {rec}")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # 保存 last
        torch.save(model.state_dict(), run_dir / "last.pth")

        # 保存 best（按 MAE 速度）
        if val_metrics["mae_speed_mpm"] < best_metric:
            best_metric = val_metrics["mae_speed_mpm"]
            torch.save(model.state_dict(), run_dir / "best.pth")
            print(f"[epoch {epoch}] new best MAE speed = {best_metric:.4f}")


if __name__ == "__main__":
    main()
