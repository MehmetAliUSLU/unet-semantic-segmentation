"""Loss functions for multi-class segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft multi-class Dice loss, averaged over classes."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        probs = logits.softmax(dim=1)
        one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dims)
        cardinality = probs.sum(dims) + one_hot.sum(dims)
        dice = (2 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1 - dice.mean()


class DiceCELoss(nn.Module):
    """Weighted sum of cross-entropy (pixel accuracy) and Dice (region overlap)."""

    def __init__(self, ce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, target):
        return (self.ce_weight * self.ce(logits, target)
                + self.dice_weight * self.dice(logits, target))
