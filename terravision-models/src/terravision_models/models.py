"""U-Net based change-detection networks.

Three fusion strategies following Daudt et al. (ICIP 2018):

* ``fc-ef``: early fusion; the two dates are concatenated channel-wise before one encoder.
* ``fc-siam-conc``: a shared encoder processes each date; decoder skips are concatenated.
* ``fc-siam-diff``: a shared encoder; decoder skips are absolute feature differences.

All variants output one logit per pixel (positive = change).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

KINDS = ("fc-ef", "fc-siam-conc", "fc-siam-diff")


class ConvBlock(nn.Sequential):
    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out),
            nn.ReLU(True),
        )


class Encoder(nn.Module):
    """Conv blocks with 2x max-pooling in between. ``forward`` returns all level features."""

    def __init__(self, in_ch: int, widths: tuple[int, ...] = (16, 32, 64)) -> None:
        super().__init__()
        self.widths = widths
        chans = (in_ch, *widths)
        self.blocks = nn.ModuleList(ConvBlock(chans[i], chans[i + 1]) for i in range(len(widths)))
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: Tensor) -> list[Tensor]:
        feats = []
        for i, block in enumerate(self.blocks):
            x = block(x if i == 0 else self.pool(x))
            feats.append(x)
        return feats


class Decoder(nn.Module):
    def __init__(self, widths: tuple[int, ...], mult: int) -> None:
        super().__init__()
        self.ups = nn.ModuleList()
        self.blocks = nn.ModuleList()
        c = widths[-1] * mult
        for w in reversed(widths[:-1]):
            self.ups.append(nn.ConvTranspose2d(c, w, 2, stride=2))
            self.blocks.append(ConvBlock(w + w * mult, w))
            c = w
        self.head = nn.Conv2d(widths[0], 1, 1)

    def forward(self, feats: list[Tensor]) -> Tensor:
        x = feats[-1]
        for up, block, skip in zip(self.ups, self.blocks, reversed(feats[:-1]), strict=True):
            x = block(torch.cat([up(x), skip], dim=1))
        return self.head(x).squeeze(1)


class ChangeNet(nn.Module):
    def __init__(self, kind: str, in_ch: int, widths: tuple[int, ...] = (16, 32, 64)) -> None:
        super().__init__()
        if kind not in KINDS:
            raise ValueError(f"unknown kind {kind!r}")
        self.kind = kind
        self.stride = 2 ** (len(widths) - 1)
        self.encoder = Encoder(2 * in_ch if kind == "fc-ef" else in_ch, widths)
        self.decoder = Decoder(widths, 2 if kind == "fc-siam-conc" else 1)

    def forward(self, before: Tensor, after: Tensor) -> Tensor:
        if before.shape[-1] % self.stride or before.shape[-2] % self.stride:
            raise ValueError(f"tile sides must be divisible by {self.stride}")
        if self.kind == "fc-ef":
            feats = self.encoder(torch.cat([before, after], dim=1))
        else:
            fa, fb = self.encoder(before), self.encoder(after)
            if self.kind == "fc-siam-conc":
                feats = [torch.cat([a, b], dim=1) for a, b in zip(fa, fb, strict=True)]
            else:
                feats = [(a - b).abs() for a, b in zip(fa, fb, strict=True)]
        return self.decoder(feats)
