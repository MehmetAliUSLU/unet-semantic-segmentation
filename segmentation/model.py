"""U-Net architecture for semantic segmentation (Ronneberger et al., 2015)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv3x3 -> BatchNorm -> ReLU) x 2"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """MaxPool 2x2 followed by DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    """Upsample, concatenate the skip connection, then DoubleConv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Pad in case the input size is not divisible by 2^depth
        dy, dx = skip.size(2) - x.size(2), skip.size(3) - x.size(3)
        if dy or dx:
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    """
    Args:
        in_channels: number of input channels (3 for RGB).
        num_classes: number of segmentation classes.
        base_channels: width of the first encoder stage; doubled at each level.
        depth: number of down-sampling stages.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 3,
                 base_channels: int = 32, depth: int = 4):
        super().__init__()
        chs = [base_channels * 2 ** i for i in range(depth + 1)]

        self.stem = DoubleConv(in_channels, chs[0])
        self.downs = nn.ModuleList(Down(chs[i], chs[i + 1]) for i in range(depth))
        self.ups = nn.ModuleList(
            Up(chs[i + 1], chs[i], chs[i]) for i in reversed(range(depth))
        )
        self.head = nn.Conv2d(chs[0], num_classes, kernel_size=1)

    def forward(self, x):
        skips = [self.stem(x)]
        for down in self.downs:
            skips.append(down(skips[-1]))

        x = skips.pop()
        for up in self.ups:
            x = up(x, skips.pop())
        return self.head(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
