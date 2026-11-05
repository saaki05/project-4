"""Self-supervised pretraining of the change-detection encoder.

Two ways to form positive pairs for the NT-Xent loss (Chen et al., SimCLR):

* ``augment``: two random augmentations of the same single-date tile;
* ``temporal``: the same location at the two dates, which is the same scene except where it
  changed (changes cover a small fraction of tiles), plus augmentation.

No change labels are used.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from terravision_models.models import Encoder


def nt_xent(z1: Tensor, z2: Tensor, temperature: float = 0.2) -> Tensor:
    """Normalised temperature-scaled cross entropy over 2N views."""
    z = nn.functional.normalize(torch.cat([z1, z2]), dim=1)
    n = z1.shape[0]
    sim = z @ z.T / temperature
    sim = sim.masked_fill(torch.eye(2 * n, dtype=torch.bool, device=z.device), float("-inf"))
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return nn.functional.cross_entropy(sim, targets)


def augment(x: Tensor, generator: torch.Generator | None = None) -> Tensor:
    """Random flips, 90-degree rotations, per-sample gain/offset jitter and noise."""
    n = x.shape[0]
    out = x
    if torch.rand(1, generator=generator).item() < 0.5:
        out = out.flip(-1)
    if torch.rand(1, generator=generator).item() < 0.5:
        out = out.flip(-2)
    out = torch.rot90(out, int(torch.randint(0, 4, (1,), generator=generator)), dims=(-2, -1))
    gain = 1 + 0.2 * (torch.rand(n, 1, 1, 1, generator=generator) - 0.5)
    offset = 0.1 * (torch.rand(n, 1, 1, 1, generator=generator) - 0.5)
    noise = 0.02 * torch.randn(out.shape, generator=generator)
    return out * gain.to(x.device) + offset.to(x.device) + noise.to(x.device)


class SSLNet(nn.Module):
    def __init__(self, encoder: Encoder, proj_dim: int = 64) -> None:
        super().__init__()
        self.encoder = encoder
        c = encoder.widths[-1]
        self.proj = nn.Sequential(nn.Linear(c, c), nn.ReLU(True), nn.Linear(c, proj_dim))

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(self.encoder(x)[-1].mean(dim=(2, 3)))


def pretrain(encoder: Encoder, before: Tensor, after: Tensor | None = None, mode: str = "augment",
             epochs: int = 20, batch: int = 64, lr: float = 1e-3, temperature: float = 0.2,
             seed: int = 0, device: str | torch.device = "cpu") -> list[float]:
    """Contrastively pretrain ``encoder`` in place; returns the per-epoch mean loss."""
    if mode == "temporal" and after is None:
        raise ValueError("temporal mode needs the 'after' tiles")
    if mode not in ("augment", "temporal"):
        raise ValueError("mode must be 'augment' or 'temporal'")
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    net = SSLNet(encoder).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    pool = before if after is None else torch.cat([before, after])
    losses = []
    for _ in range(epochs):
        net.train()
        order = torch.randperm(len(before), generator=g)
        total, nb = 0.0, 0
        for i in range(0, len(order) - 1, batch):
            idx = order[i:i + batch]
            if len(idx) < 2:
                continue
            if mode == "temporal":
                a, b = before[idx], after[idx]
            else:
                pick = torch.randint(0, len(pool), (len(idx),), generator=g)
                a = b = pool[pick]
            v1, v2 = augment(a, g).to(device), augment(b, g).to(device)
            loss = nt_xent(net(v1), net(v2), temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total, nb = total + loss.item(), nb + 1
        losses.append(total / max(nb, 1))
    encoder.cpu()
    return losses
