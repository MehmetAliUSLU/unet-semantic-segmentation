"""Confusion-matrix based segmentation metrics."""

import torch


class SegmentationMetrics:
    """Accumulates a confusion matrix over batches and derives IoU / Dice / accuracy."""

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.confusion = torch.zeros(self.num_classes, self.num_classes, dtype=torch.int64)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor):
        pred = logits.argmax(dim=1).flatten().cpu()
        target = target.flatten().cpu()
        idx = target * self.num_classes + pred
        self.confusion += torch.bincount(
            idx, minlength=self.num_classes ** 2
        ).view(self.num_classes, self.num_classes)

    def compute(self) -> dict:
        cm = self.confusion.double()
        tp = cm.diag()
        fp = cm.sum(0) - tp
        fn = cm.sum(1) - tp
        iou = tp / (tp + fp + fn).clamp(min=1)
        dice = 2 * tp / (2 * tp + fp + fn).clamp(min=1)
        return {
            "pixel_acc": (tp.sum() / cm.sum().clamp(min=1)).item(),
            "mean_iou": iou.mean().item(),
            "mean_dice": dice.mean().item(),
            "per_class_iou": iou.tolist(),
        }
