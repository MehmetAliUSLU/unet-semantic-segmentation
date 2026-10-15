"""
Train a U-Net for semantic segmentation.

Examples:
    python train.py --dataset pets --epochs 25
    python train.py --dataset shapes --epochs 10      # no download needed
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from segmentation import DiceCELoss, SegmentationMetrics, UNet
from segmentation.data import PetSegmentation, SyntheticShapes
from segmentation.model import count_parameters
from segmentation.visualize import save_prediction_grid, save_training_curves


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["pets", "shapes"], default="pets")
    p.add_argument("--data-root", default="data")
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0,
                   help="use only the first N samples of each split (quick debugging)")
    p.add_argument("--out", default="runs/exp")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    return p.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_datasets(args):
    if args.dataset == "pets":
        train_ds = PetSegmentation(args.data_root, "trainval", args.img_size, train=True)
        val_ds = PetSegmentation(args.data_root, "test", args.img_size, train=False)
        classes = PetSegmentation.classes
    else:
        train_ds = SyntheticShapes(length=2000, size=args.img_size, seed=0)
        val_ds = SyntheticShapes(length=300, size=args.img_size, seed=1)
        classes = SyntheticShapes.classes

    if args.limit:
        train_ds = torch.utils.data.Subset(train_ds, range(min(args.limit, len(train_ds))))
        val_ds = torch.utils.data.Subset(val_ds, range(min(args.limit, len(val_ds))))
    return train_ds, val_ds, classes


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None,
              metrics=None, amp=True):
    training = optimizer is not None
    model.train(training)
    total_loss, n = 0.0, 0
    with torch.set_grad_enabled(training):
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                logits = model(images)
                loss = criterion(logits.float(), masks)

            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            if metrics is not None:
                metrics.update(logits, masks)

            total_loss += loss.item() * images.size(0)
            n += images.size(0)
    return total_loss / max(n, 1)


def main():
    args = parse_args()
    seed_everything(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not args.no_amp

    train_ds, val_ds, classes = build_datasets(args)
    loader_kw = dict(batch_size=args.batch_size, num_workers=args.workers,
                     pin_memory=device.type == "cuda",
                     persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)

    model = UNet(3, len(classes), base_channels=args.base_channels).to(device)
    print(f"Device: {device} | AMP: {amp} | params: {count_parameters(model) / 1e6:.2f}M")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Classes: {classes}")

    criterion = DiceCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs,
        steps_per_epoch=1, pct_start=0.1)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    metrics = SegmentationMetrics(len(classes))

    config = {**vars(args), "classes": list(classes)}
    (out / "config.json").write_text(json.dumps(config, indent=2))

    history, best_miou = [], -1.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, criterion, device,
                               optimizer, scaler, amp=amp)
        metrics.reset()
        val_loss = run_epoch(model, val_loader, criterion, device,
                             metrics=metrics, amp=amp)
        scheduler.step()

        m = metrics.compute()
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, **m})
        print(f"[{epoch:02d}/{args.epochs}] train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} mIoU={m['mean_iou']:.4f} "
              f"dice={m['mean_dice']:.4f} acc={m['pixel_acc']:.4f} "
              f"({time.time() - t0:.0f}s)")

        if m["mean_iou"] > best_miou:
            best_miou = m["mean_iou"]
            torch.save({"model": model.state_dict(), "config": config,
                        "epoch": epoch, "metrics": m}, out / "best.pt")

    (out / "history.json").write_text(json.dumps(history, indent=2))
    save_training_curves(history, out / "curves.png")

    # Visualize predictions of the best checkpoint on a few validation samples
    model.load_state_dict(torch.load(out / "best.pt", map_location=device)["model"])
    model.eval()
    images, masks = next(iter(DataLoader(val_ds, batch_size=4, shuffle=True)))
    with torch.no_grad():
        preds = model(images.to(device)).argmax(1)
    save_prediction_grid(images, masks, preds, out / "predictions.png", list(classes))

    best = max(history, key=lambda h: h["mean_iou"])
    print(f"\nBest epoch {best['epoch']}: mIoU={best['mean_iou']:.4f} "
          f"per-class IoU={dict(zip(classes, map(lambda v: round(v, 4), best['per_class_iou'])))}")
    print(f"Artifacts saved to {out.resolve()}")


if __name__ == "__main__":
    main()
