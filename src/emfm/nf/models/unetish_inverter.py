from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1, groups: int = 8):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        # GroupNorm is stable for small batch sizes (common in EM field problems)
        g = max(1, min(groups, out_ch))
        self.gn = nn.GroupNorm(g, out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.conv(x)))

class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = ConvGNAct(ch, ch, 3, 1, 1)
        self.c2 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False)
        self.gn = nn.GroupNorm(max(1, min(8, ch)), ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.c1(x)
        y = self.gn(self.c2(y))
        return self.act(x + y)

class UNetishInverter(nn.Module):
    """Baseline inverter: [B,4,11,11] -> [B,12,51,51]

    Why this design:
    - Input grid is small (11x11) so heavy downsampling isn't useful.
    - We encode local patterns, then upsample to target resolution (51x51),
      then refine with several residual blocks.
    - This is a strong baseline and trains fast.
    """

    def __init__(self, in_ch: int = 4, out_ch: int = 12, base: int = 64, n_res: int = 6,
                 add_xy_positional: bool = True):
        super().__init__()
        self.add_xy_positional = add_xy_positional
        pos_ch = 2 if add_xy_positional else 0

        self.stem = nn.Sequential(
            ConvGNAct(in_ch + pos_ch, base, 3, 1, 1),
            ResBlock(base),
            ResBlock(base),
        )

        self.enc2 = nn.Sequential(
            ConvGNAct(base, base * 2, 3, 1, 1),
            ResBlock(base * 2),
        )

        self.enc3 = nn.Sequential(
            ConvGNAct(base * 2, base * 2, 3, 1, 1),
            ResBlock(base * 2),
        )

        self.refine = nn.Sequential(*[ResBlock(base * 2) for _ in range(n_res)])

        self.head = nn.Sequential(
            ConvGNAct(base * 2, base, 3, 1, 1),
            nn.Conv2d(base, out_ch, 1, 1, 0)
        )

    def _make_xy(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,H,W]
        B, _, H, W = x.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, H, device=x.device, dtype=x.dtype),
            torch.linspace(-1.0, 1.0, W, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        xy = torch.stack([xx, yy], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)  # [B,2,H,W]
        return xy

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,4,11,11]
        if self.add_xy_positional:
            x = torch.cat([x, self._make_xy(x)], dim=1)

        h = self.stem(x)
        h = self.enc2(h)
        h = self.enc3(h)

        # upsample to 51x51
        h = F.interpolate(h, size=(51, 51), mode="bilinear", align_corners=False)

        h = self.refine(h)
        y = self.head(h)
        return y
