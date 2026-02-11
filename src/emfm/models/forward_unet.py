from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv(in_ch, out_ch, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p),
        nn.BatchNorm2d(out_ch),
        nn.SiLU(inplace=True),
    )

class ForwardUNetLite(nn.Module):
    """
    Input:  [B, Cin, 11, 11]
    Output: [B, Cout, 51, 51]
    Strategy:
      1) encode at 11x11
      2) upsample to 51x51
      3) refine with a few conv blocks
    """
    def __init__(self, cin: int = 4, cout: int = 12, width: int = 64):
        super().__init__()
        self.enc1 = nn.Sequential(conv(cin, width), conv(width, width))
        self.enc2 = nn.Sequential(conv(width, width*2, s=2, p=1), conv(width*2, width*2))  # ~6x6
        self.bott = nn.Sequential(conv(width*2, width*2), conv(width*2, width*2))

        self.proj = nn.Conv2d(width*2, width, kernel_size=1)
        self.refine = nn.Sequential(
            conv(width, width),
            conv(width, width),
            nn.Conv2d(width, cout, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)              # [B,w,11,11]
        x2 = self.enc2(x1)             # [B,2w,~6,~6]
        b  = self.bott(x2)             # [B,2w,~6,~6]
        u  = F.interpolate(b, size=(51, 51), mode="bilinear", align_corners=False)
        u  = self.proj(u)              # [B,w,51,51]
        y  = self.refine(u)            # [B,cout,51,51]
        return y
