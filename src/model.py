"""
EEGNet (Lawhern et al., 2018) implemented in PyTorch.

Reference: "EEGNet: A Compact Convolutional Neural Network for EEG-based
Brain-Computer Interfaces" - https://arxiv.org/abs/1611.08024

Input shape expected: (batch, 1, channels, samples)
    - channels = number of EEG electrodes (22 for BCI IV 2a)
    - samples  = number of time points per trial (e.g. 4s * 250Hz = 1000)
"""

import torch
import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(
        self,
        n_classes: int = 4,
        channels: int = 22,
        samples: int = 1000,
        dropout_rate: float = 0.5,
        kernel_length: int = 64,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
    ):
        super().__init__()
        self.channels = channels
        self.samples = samples

        # Block 1: temporal convolution -> depthwise spatial convolution
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length), padding="same", bias=False),
            nn.BatchNorm2d(F1),
            # depthwise conv across all channels -> learns spatial filters per temporal filter
            nn.Conv2d(F1, F1 * D, (channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate),
        )

        # Block 2: separable convolution (depthwise + pointwise)
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding="same", groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate),
        )

        # Figure out the flattened feature size with a dummy forward pass. Use
        # eval() here, not just no_grad() - no_grad() only disables gradient
        # tracking, it doesn't stop BatchNorm from updating its running stats or
        # Dropout from consuming randomness, so without eval() this dummy pass
        # would quietly perturb BatchNorm's running mean/var using all-zero input
        # before real training even starts.
        was_training = self.training
        self.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 1, channels, samples)
            feat_dim = self.block2(self.block1(dummy)).flatten(1).shape[1]
        self.train(was_training)

        self.classify = nn.Linear(feat_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, samples) or (batch, 1, channels, samples)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = x.flatten(1)
        return self.classify(x)


if __name__ == "__main__":
    # quick shape sanity check
    model = EEGNet(n_classes=4, channels=22, samples=1000)
    x = torch.randn(8, 22, 1000)
    out = model(x)
    print("output shape:", out.shape)  # expect (8, 4)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {n_params:,}")
