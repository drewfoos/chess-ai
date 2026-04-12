"""Chess neural network: residual tower with SE layers, policy, value, and moves-left heads.

Architecture (matching Leela Chess Zero):
    Input: 112 × 8 × 8
    → Initial Conv(3×3, filters) + BN + Mish
    → N × ResidualBlock(3×3 conv + BN + Mish + 3×3 conv + BN + SE + skip + Mish)
    → Policy Head (attention): tokens → embed → Q/K projections → Q@K^T/√d → promotions → map to 1858
    → Policy Head (classical): Conv(1×1, 80) + BN + Mish + Flatten + FC(5120 → 1858)
    → Value Head: Conv(1×1, 32) + BN + Mish + Flatten + FC(2048 → 128) + Mish + FC(128 → 3)
    → Moves-Left Head: Conv(1×1, 8) + BN + Mish + Flatten + FC(512 → 128) + Mish + FC(128 → 1) + ReLU
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from training.config import NetworkConfig
from training.encoder import POLICY_SIZE, index_to_move, rank_of, file_of


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
        out = F.mish(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = F.mish(out + residual)
        return out


def _build_attention_policy_index() -> torch.Tensor:
    """Build index map from 1858 policy indices to 4288-dim attention output.

    The attention policy head produces:
        - 64×64 = 4096 from-to logits (normal moves + knight promotions)
        - 8×24 = 192 promotion logits (Q/R/B for each from-file × to-file)
    Concatenated to 4288. This function maps each of the 1858 policy indices to
    its source position in that 4288-dim vector.

    Convention: knight promotion uses the base 64×64 entry; Q/R/B promotions
    use the 192 promotion section (which already includes base + offset).
    """
    index_map = torch.zeros(POLICY_SIZE, dtype=torch.long)

    for i in range(POLICY_SIZE):
        from_sq, to_sq, promo = index_to_move(i)

        if promo is None:
            # Check if this is a queen promotion (from rank 6 to rank 7)
            if rank_of(from_sq) == 6 and rank_of(to_sq) == 7:
                # Queen promotion → promotion section, promo_type=0 (Q)
                from_file = file_of(from_sq)
                to_file = file_of(to_sq)
                index_map[i] = 4096 + from_file * 24 + to_file * 3 + 0
            else:
                # Normal move → 64×64 section
                index_map[i] = from_sq * 64 + to_sq
        elif promo == 'n':
            # Knight promotion → base 64×64 entry
            index_map[i] = from_sq * 64 + to_sq
        elif promo == 'r':
            # Rook promotion → promotion section, promo_type=1
            from_file = file_of(from_sq)
            to_file = file_of(to_sq)
            index_map[i] = 4096 + from_file * 24 + to_file * 3 + 1
        elif promo == 'b':
            # Bishop promotion → promotion section, promo_type=2
            from_file = file_of(from_sq)
            to_file = file_of(to_sq)
            index_map[i] = 4096 + from_file * 24 + to_file * 3 + 2

    return index_map


class AttentionPolicyHead(nn.Module):
    """Attention-based policy head (Lc0 style).

    Reshapes the body output into 64 square tokens, projects to Q and K vectors,
    computes Q@K^T scaled dot-product as from-to move logits, handles promotions
    with learned offsets, and gathers into the 1858-dim policy vector.
    """

    def __init__(self, body_channels: int, embedding_size: int = 64, d_model: int = 64):
        super().__init__()
        self.d_model = d_model
        self.scale = math.sqrt(d_model)

        # Token embedding: project each square's features
        self.embedding = nn.Linear(body_channels, embedding_size)

        # Q and K projections
        self.query = nn.Linear(embedding_size, d_model, bias=False)
        self.key = nn.Linear(embedding_size, d_model, bias=False)

        # Promotion offsets: 4 scalars per promotion-rank square [Q, R, B, N]
        self.promotion = nn.Linear(d_model, 4, bias=False)

        # Register the policy index map as a buffer (moves with the model to GPU, saved in state)
        self.register_buffer('policy_index', _build_attention_policy_index())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)

        # (B, C, 8, 8) → (B, 64, C) → (B, 64, embed)
        tokens = x.permute(0, 2, 3, 1).reshape(B, 64, -1)
        tokens = F.selu(self.embedding(tokens))

        # Q and K projections
        queries = self.query(tokens)   # (B, 64, d)
        keys = self.key(tokens)        # (B, 64, d)

        # Scaled dot-product attention logits (no softmax — these are policy logits)
        # (B, 64, d) @ (B, d, 64) → (B, 64, 64)
        attn_logits = torch.bmm(queries, keys.transpose(1, 2)) / self.scale

        # Promotion handling: extract last-rank keys and compute offsets
        # Squares 56-63 = rank 7 (promotion target rank)
        promo_keys = keys[:, -8:, :]                     # (B, 8, d)
        promo_offsets = self.promotion(promo_keys)        # (B, 8, 4) — [Q, R, B, N]
        promo_offsets = promo_offsets.permute(0, 2, 1)    # (B, 4, 8_to)
        promo_offsets = promo_offsets * self.scale         # Scale to match pre-division logits

        # Knight offset is the base; Q/R/B offsets add to it
        # promo_offsets[:, 0:3, :] are [Q, R, B], promo_offsets[:, 3:4, :] is [N]
        promo_offsets = promo_offsets[:, :3, :] + promo_offsets[:, 3:4, :]  # (B, 3, 8_to)

        # Base knight promotion logits: from rank 6 (squares 48-55) to rank 7 (squares 56-63)
        n_promo = attn_logits[:, -16:-8, -8:]  # (B, 8_from, 8_to)

        # Q/R/B promotion logits = base + offset (offset broadcasts over from-files)
        q_promo = n_promo + promo_offsets[:, 0:1, :]  # (B, 8, 8)
        r_promo = n_promo + promo_offsets[:, 1:2, :]  # (B, 8, 8)
        b_promo = n_promo + promo_offsets[:, 2:3, :]  # (B, 8, 8)

        # Stack and reshape: (B, 8_from, 8_to, 3_promo) → (B, 8, 24) → (B, 192)
        promo_logits = torch.stack([q_promo, r_promo, b_promo], dim=3)  # (B, 8, 8, 3)
        promo_logits = promo_logits.reshape(B, -1)  # (B, 192)

        # Concatenate 64×64 and promotions: (B, 4096 + 192) = (B, 4288)
        full_logits = torch.cat([attn_logits.reshape(B, -1), promo_logits], dim=1)

        # Gather into 1858-dim policy vector using pre-built index map
        policy = full_logits[:, self.policy_index]  # (B, 1858)

        return policy


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
        if config.use_attention_policy:
            self.policy_head = AttentionPolicyHead(
                body_channels=nf,
                embedding_size=config.policy_embedding_size,
                d_model=config.policy_d_model,
            )
        else:
            self.policy_conv = nn.Conv2d(nf, config.policy_conv_filters, 1, bias=False)
            self.policy_bn = nn.BatchNorm2d(config.policy_conv_filters)
            self.policy_fc = nn.Linear(config.policy_conv_filters * 64, config.policy_size)

        # Value head (WDL)
        self.value_conv = nn.Conv2d(nf, config.value_conv_filters, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(config.value_conv_filters)
        self.value_fc1 = nn.Linear(config.value_conv_filters * 64, config.value_fc_size)
        self.value_fc2 = nn.Linear(config.value_fc_size, config.value_size)

        # Moves-left head
        self.mlh_conv = nn.Conv2d(nf, config.mlh_conv_filters, 1, bias=False)
        self.mlh_bn = nn.BatchNorm2d(config.mlh_conv_filters)
        self.mlh_fc1 = nn.Linear(config.mlh_conv_filters * 64, config.mlh_fc_size)
        self.mlh_fc2 = nn.Linear(config.mlh_fc_size, 1)

        self._init_weights()

    def _init_weights(self):
        """Apply Glorot (Xavier) normal initialization to all layers."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Initial conv
        x = F.mish(self.input_bn(self.input_conv(x)))

        # Residual tower
        for block in self.blocks:
            x = block(x)

        # Policy head
        if self.config.use_attention_policy:
            p = self.policy_head(x)
        else:
            p = F.mish(self.policy_bn(self.policy_conv(x)))
            p = p.reshape(p.size(0), -1)
            p = self.policy_fc(p)  # Raw logits

        # Value head (WDL)
        v = F.mish(self.value_bn(self.value_conv(x)))
        v = v.reshape(v.size(0), -1)
        v = F.mish(self.value_fc1(v))
        v = self.value_fc2(v)  # Raw WDL logits (softmax applied in loss / at inference)

        # Moves-left head
        m = F.mish(self.mlh_bn(self.mlh_conv(x)))
        m = m.reshape(m.size(0), -1)
        m = F.mish(self.mlh_fc1(m))
        m = F.relu(self.mlh_fc2(m))  # ReLU: moves left is non-negative
        m = m.squeeze(1)  # (B,)

        return p, v, m
