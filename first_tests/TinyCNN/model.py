"""
TinyCNN: Micro-scale fire/smoke/other classifier for Arduino Nano 33 BLE.

Architecture: Depthwise-separable convolutions (MobileNet-style)
  stem:   Conv(3→32, stride=2)    → 48×48×32
  block1: DWSep(32→64, stride=1)  → 48×48×64
  block2: DWSep(64→128, stride=2) → 24×24×128
  block3: DWSep(128→128,stride=1) → 24×24×128
  block4: DWSep(128→256,stride=2) → 12×12×256
  block5: DWSep(256→256,stride=1) → 12×12×256
  GAP → Linear(256, num_classes)

~134 K parameters → ~134 kB when exported as INT8 TFLite (well under 500 kB).

Classes (matching first_tests/YOLOv26/best.pt):
  0 = fire
  1 = other
  2 = smoke
"""

import torch
import torch.nn as nn


CLASS_NAMES = ["fire", "other", "smoke"]
INPUT_SIZE = 96  # pixels (square)


class DWSepBlock(nn.Module):
    """Depthwise-separable convolution block.

    DW(3×3) → BN → ReLU6 → PW(1×1) → BN → ReLU6
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1,
                      groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU6(inplace=True),
        )
        self.pw = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class TinyCNN(nn.Module):
    """Tiny depthwise-separable CNN for 3-class image classification."""

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )
        self.blocks = nn.Sequential(
            DWSepBlock(32, 64, stride=1),
            DWSepBlock(64, 128, stride=2),
            DWSepBlock(128, 128, stride=1),
            DWSepBlock(128, 256, stride=2),
            DWSepBlock(256, 256, stride=1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = TinyCNN()
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    out = model(dummy)
    n_params = count_parameters(model)
    print(f"Output shape : {out.shape}")
    print(f"Parameters   : {n_params:,}")
    print(f"INT8 size est: ~{n_params / 1024:.0f} kB  (target < 500 kB)")
