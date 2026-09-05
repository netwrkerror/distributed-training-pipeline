"""A deliberately small convolutional classifier.

The brief says "ResNet-18 scale or smaller" and this is decisively smaller, for a
reason that matters more than accuracy: **milestone B cannot measure an input
pipeline that a heavy model is hiding.** If the forward and backward pass dominate
step time, the DataLoader always looks fast enough, the num_workers sweep in B1 is
flat, and there is no knee to find in B2. A cheap model keeps data loading on the
critical path, which is where this repo needs it.

It is also run four times concurrently on CPU from A4 onward, so every millisecond
here is multiplied by the number of experiments still to come.

Note on BatchNorm and DDP: under DistributedDataParallel each rank normalises using
its *own* local batch statistics - the running stats are not synchronised across
ranks by default. With per-device batches this size that is standard and fine.
torch.nn.SyncBatchNorm exists for when local batches get small enough that per-rank
statistics become noisy, which is a GPU-era concern (milestone C), not one for A4.
"""

from __future__ import annotations

import torch
from torch import nn


def _block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    """3x64x64 -> num_classes. Roughly 0.4M parameters."""

    def __init__(self, num_classes: int, in_channels: int = 3, width: int = 32) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _block(in_channels, width),  # 64 -> 32
            _block(width, width * 2),  # 32 -> 16
            _block(width * 2, width * 4),  # 16 -> 8
            _block(width * 4, width * 4),  # 8  -> 4
        )
        # Global average pooling rather than a flatten: it makes the head independent
        # of input resolution, so changing crop_size does not silently break the model.
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(width * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
