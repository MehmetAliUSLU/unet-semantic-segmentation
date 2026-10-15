"""Datasets and joint image/mask transforms."""

import random

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import OxfordIIITPet

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

PET_CLASSES = ("pet", "background", "border")
SHAPE_CLASSES = ("background", "circle", "rectangle")


class SegTransform:
    """Applies the same geometric transform to an image and its mask."""

    def __init__(self, size: int = 256, train: bool = True):
        self.size = size
        self.train = train

    def __call__(self, image: Image.Image, mask: Image.Image):
        image = TF.resize(image, [self.size, self.size],
                          interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.size, self.size],
                         interpolation=TF.InterpolationMode.NEAREST)

        if self.train:
            if random.random() < 0.5:
                image, mask = TF.hflip(image), TF.hflip(mask)
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST,
                             fill=1)  # 1 -> "background" after the -1 shift below
            image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
            image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))

        image = TF.normalize(TF.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        return image, mask


class PetSegmentation(Dataset):
    """
    Oxford-IIIT Pet trimaps: 1 = pet, 2 = background, 3 = border.
    Labels are shifted to 0..2 so they can be used directly with CrossEntropyLoss.
    """

    classes = PET_CLASSES

    def __init__(self, root: str, split: str = "trainval", size: int = 256,
                 train: bool = True, download: bool = True):
        self.ds = OxfordIIITPet(root, split=split, target_types="segmentation",
                                download=download)
        self.transform = SegTransform(size, train)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        image, mask = self.ds[idx]
        image, mask = self.transform(image.convert("RGB"), mask)
        return image, mask - 1


class SyntheticShapes(Dataset):
    """
    Procedurally generated images with circles and rectangles on a noisy background.
    Needs no download, which makes it handy for quick experiments and CI.
    """

    classes = SHAPE_CLASSES

    def __init__(self, length: int = 512, size: int = 128, seed: int = 0):
        self.length = length
        self.size = size
        self.seed = seed

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        rng = np.random.default_rng(self.seed * 100_003 + idx)
        s = self.size
        image = rng.normal(0.5, 0.08, (s, s, 3)).astype(np.float32)
        image += rng.uniform(-0.2, 0.2, 3).astype(np.float32)  # background tint
        mask = np.zeros((s, s), dtype=np.int64)
        yy, xx = np.mgrid[:s, :s]

        for _ in range(rng.integers(1, 5)):
            color = rng.uniform(0, 1, 3).astype(np.float32)
            if rng.random() < 0.5:
                cy, cx = rng.integers(0, s, 2)
                r = rng.integers(s // 12, s // 5)
                region = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
                label = 1
            else:
                y0, x0 = rng.integers(0, s - s // 6, 2)
                h, w = rng.integers(s // 8, s // 3, 2)
                region = (yy >= y0) & (yy < y0 + h) & (xx >= x0) & (xx < x0 + w)
                label = 2
            image[region] = color + rng.normal(0, 0.03, (region.sum(), 3))
            mask[region] = label

        image = torch.from_numpy(np.clip(image, 0, 1)).permute(2, 0, 1)
        image = TF.normalize(image, IMAGENET_MEAN, IMAGENET_STD)
        return image, torch.from_numpy(mask)


def denormalize(image: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=image.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=image.device).view(3, 1, 1)
    return (image * std + mean).clamp(0, 1)
