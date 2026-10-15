"""
Run a trained U-Net on your own images.

Example:
    python predict.py --checkpoint runs/pets/best.pt --input my_cat.jpg --out outputs/
    python predict.py --checkpoint runs/pets/best.pt --input folder_of_images/
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

from segmentation import UNet
from segmentation.data import IMAGENET_MEAN, IMAGENET_STD
from segmentation.visualize import colorize, overlay

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_model(checkpoint: str, device):
    ckpt = torch.load(checkpoint, map_location=device)
    cfg = ckpt["config"]
    model = UNet(3, len(cfg["classes"]), base_channels=cfg["base_channels"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), cfg


@torch.no_grad()
def segment(model, image: Image.Image, size: int, device) -> np.ndarray:
    """Returns an HxW label map at the original image resolution."""
    x = TF.resize(image, [size, size])
    x = TF.normalize(TF.to_tensor(x), IMAGENET_MEAN, IMAGENET_STD).unsqueeze(0).to(device)
    logits = model(x)
    logits = F.interpolate(logits, size=(image.height, image.width),
                           mode="bilinear", align_corners=False)
    return logits.argmax(1)[0].cpu().numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input", required=True, help="image file or folder")
    p.add_argument("--out", default="outputs")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(args.checkpoint, device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    src = Path(args.input)
    files = sorted(f for f in src.iterdir() if f.suffix.lower() in IMAGE_EXTS) \
        if src.is_dir() else [src]

    for f in files:
        image = Image.open(f).convert("RGB")
        mask = segment(model, image, cfg["img_size"], device)
        img_np = np.asarray(image, dtype=np.float32) / 255.0
        Image.fromarray(colorize(mask)).save(out / f"{f.stem}_mask.png")
        Image.fromarray((overlay(img_np, mask) * 255).astype(np.uint8)).save(
            out / f"{f.stem}_overlay.png")
        share = {c: f"{(mask == i).mean() * 100:.1f}%" for i, c in enumerate(cfg["classes"])}
        print(f"{f.name}: {share}")

    print(f"Saved {len(files)} result(s) to {out.resolve()}")


if __name__ == "__main__":
    main()
