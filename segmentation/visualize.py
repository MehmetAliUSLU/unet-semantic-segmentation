"""Plotting helpers for predictions and training curves."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import denormalize

PALETTE = np.array([
    [230, 25, 75], [60, 60, 60], [255, 225, 25], [60, 180, 75],
    [0, 130, 200], [245, 130, 48], [145, 30, 180], [70, 240, 240],
], dtype=np.uint8)


def colorize(mask: np.ndarray) -> np.ndarray:
    return PALETTE[mask % len(PALETTE)]


def overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """image: HxWx3 float in [0, 1]; mask: HxW int."""
    return (1 - alpha) * image + alpha * colorize(mask) / 255.0


def save_prediction_grid(images, targets, preds, path, class_names=None, max_rows=4):
    rows = min(max_rows, len(images))
    fig, axes = plt.subplots(rows, 4, figsize=(12, 3 * rows), squeeze=False)
    titles = ["Image", "Ground truth", "Prediction", "Overlay"]
    for r in range(rows):
        img = denormalize(images[r]).permute(1, 2, 0).cpu().numpy()
        gt = targets[r].cpu().numpy()
        pr = preds[r].cpu().numpy()
        for c, panel in enumerate([img, colorize(gt), colorize(pr), overlay(img, pr)]):
            axes[r, c].imshow(panel)
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(titles[c])
    if class_names:
        handles = [plt.Rectangle((0, 0), 1, 1, color=PALETTE[i] / 255.0)
                   for i in range(len(class_names))]
        fig.legend(handles, class_names, loc="lower center", ncol=len(class_names))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=100)
    plt.close(fig)


def save_training_curves(history: list, path):
    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, [h["train_loss"] for h in history], label="train")
    ax1.plot(epochs, [h["val_loss"] for h in history], label="val")
    ax1.set_title("Loss (CE + Dice)")
    ax1.set_xlabel("epoch")
    ax1.legend()
    ax2.plot(epochs, [h["mean_iou"] for h in history], label="mIoU")
    ax2.plot(epochs, [h["mean_dice"] for h in history], label="mean Dice")
    ax2.plot(epochs, [h["pixel_acc"] for h in history], label="pixel acc")
    ax2.set_title("Validation metrics")
    ax2.set_xlabel("epoch")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


@torch.no_grad()
def predict_mask(model, image: torch.Tensor) -> torch.Tensor:
    """image: 3xHxW normalized tensor -> HxW predicted labels."""
    model.eval()
    device = next(model.parameters()).device
    return model(image.unsqueeze(0).to(device)).argmax(1)[0].cpu()
