"""Chess neural network: residual tower with SE layers, policy and value heads.

Architecture (matching Leela Chess Zero):
    Input: 112 × 8 × 8
    → Initial Conv(3×3, filters) + BN + ReLU
    → N × ResidualBlock(3×3 conv + BN + ReLU + 3×3 conv + BN + SE + skip + ReLU)
    → Policy Head: Conv(1×1, 80) + BN + ReLU + Flatten + FC(5120 → 1858)
    → Value Head: Conv(1×1, 32) + BN + ReLU + Flatten + FC(2048 → 128) + ReLU + FC(128 → 3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from training.config import NetworkConfig


class SqueezeExcitation(nn.Module):
    """Leela-style SE: produces both multiplicative weights and additive biases."""

    def __init__(self, channels: int, ratio: int):
        super().__init__()
        mid = channels // ratio
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, 2 * channels)
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        # Global average pooling
        z = x.mean(dim=(2, 3))  # (B, C)
        z = F.relu(self.fc1(z))
        z = self.fc2(z)  # (B, 2C)
        # Split into weights and biases
        weights, biases = z.split(self.channels, dim=1)
        weights = torch.sigmoid(weights).view(b, c, 1, 1)
        biases = biases.view(b, c, 1, 1)
        return weights * x + biases


class ResidualBlock(nn.Module):
    """Residual block with SE layer."""

    def __init__(self, channels: int, se_ratio: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SqueezeExcitation(channels, se_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = F.relu(out + residual)
        return out


class ChessNetwork(nn.Module):
    """Full chess neural network with residual tower, policy head, and WDL value head."""

    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.config = config
        nf = config.num_filters

        # Initial convolution
        self.input_conv = nn.Conv2d(config.input_planes, nf, 3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(nf)

        # Residual tower
        self.blocks = nn.ModuleList([
            ResidualBlock(nf, config.se_ratio)
            for _ in range(config.num_blocks)
        ])

        # Policy head
        self.policy_conv = nn.Conv2d(nf, config.policy_conv_filters, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(config.policy_conv_filters)
        self.policy_fc = nn.Linear(config.policy_conv_filters * 64, config.policy_size)

        # Value head (WDL)
        self.value_conv = nn.Conv2d(nf, config.value_conv_filters, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(config.value_conv_filters)
        self.value_fc1 = nn.Linear(config.value_conv_filters * 64, config.value_fc_size)
        self.value_fc2 = nn.Linear(config.value_fc_size, config.value_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Initial conv
        x = F.relu(self.input_bn(self.input_conv(x)))

        # Residual tower
        for block in self.blocks:
            x = block(x)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.reshape(p.size(0), -1)
        p = self.policy_fc(p)  # Raw logits

        # Value head (WDL)
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.reshape(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = self.value_fc2(v)  # Raw WDL logits (softmax applied in loss / at inference)

        return p, v
