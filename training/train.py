"""Training loop for the chess neural network.

Loss function:
    L = L_policy + L_value
    L_policy = cross_entropy(policy_target, policy_output)
    L_value  = cross_entropy(wdl_target, wdl_output)
    L2 regularization is handled by the optimizer (weight_decay).

Usage:
    python -m training.train --data data/synthetic.npz --epochs 10
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from training.config import NetworkConfig
from training.model import ChessNetwork
from training.dataset import ChessDataset


def compute_loss(
    policy_logits: torch.Tensor,
    value_logits: torch.Tensor,
    policy_target: torch.Tensor,
    value_target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute combined policy + value loss.

    Args:
        policy_logits: Raw logits from policy head (B, 1858)
        value_logits: Raw logits from value head (B, 3)
        policy_target: MCTS visit distribution target (B, 1858)
        value_target: WDL target (B, 3)

    Returns:
        (total_loss, policy_loss, value_loss) tuple.
    """
    # Policy loss: cross-entropy with soft targets
    log_probs = F.log_softmax(policy_logits, dim=1)
    policy_loss = -(policy_target * log_probs).sum(dim=1).mean()

    # Value loss: cross-entropy with soft WDL targets
    value_log_probs = F.log_softmax(value_logits, dim=1)
    value_loss = -(value_target * value_log_probs).sum(dim=1).mean()

    return policy_loss + value_loss, policy_loss, value_loss


def create_optimizer(model: ChessNetwork, lr: float = 1e-3, weight_decay: float = 1e-4):
    """Create AdamW optimizer with decoupled weight decay."""
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def train_step(
    model: ChessNetwork,
    optimizer: torch.optim.Optimizer,
    planes: torch.Tensor,
    policy_target: torch.Tensor,
    value_target: torch.Tensor,
) -> float:
    """Execute one training step. Returns the loss value."""
    model.train()
    optimizer.zero_grad()

    policy_logits, value_logits = model(planes)
    loss, _, _ = compute_loss(policy_logits, value_logits, policy_target, value_target)

    loss.backward()
    optimizer.step()

    return loss.item()


def train(
    data_paths: list[str],
    config: NetworkConfig = NetworkConfig(),
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    lr_milestones: list[int] | None = None,
    lr_gamma: float = 0.1,
    checkpoint_dir: str = 'checkpoints',
    device: str = 'auto',
):
    """Full training loop."""
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Device: {device}")
    print(f"Config: {config.num_blocks} blocks, {config.num_filters} filters")

    dataset = ChessDataset(data_paths)
    print(f"Training data: {len(dataset)} positions from {len(data_paths)} files")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = ChessNetwork(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    optimizer = create_optimizer(model, lr=lr, weight_decay=weight_decay)

    scheduler = None
    if lr_milestones:
        scheduler = MultiStepLR(optimizer, milestones=lr_milestones, gamma=lr_gamma)
        print(f"LR schedule: milestones={lr_milestones}, gamma={lr_gamma}")

    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        num_batches = 0
        start = time.time()

        for planes, policies, values in loader:
            planes = planes.to(device)
            policies = policies.to(device)
            values = values.to(device)

            model.train()
            optimizer.zero_grad()

            policy_logits, value_logits = model(planes)
            loss, p_loss, v_loss = compute_loss(
                policy_logits, value_logits, policies, values
            )

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_policy_loss += p_loss.item()
            epoch_value_loss += v_loss.item()
            num_batches += 1

        if scheduler:
            scheduler.step()

        elapsed = time.time() - start
        avg_loss = epoch_loss / max(num_batches, 1)
        avg_p = epoch_policy_loss / max(num_batches, 1)
        avg_v = epoch_value_loss / max(num_batches, 1)
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss: {avg_loss:.4f} (policy: {avg_p:.4f}, value: {avg_v:.4f}) | "
            f"LR: {current_lr:.6f} | Time: {elapsed:.1f}s"
        )

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            path = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'loss': avg_loss,
            }, path)
            print(f"  Saved checkpoint: {path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train chess neural network')
    parser.add_argument('--data', nargs='+', required=True, help='Training .npz files')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--blocks', type=int, default=10)
    parser.add_argument('--filters', type=int, default=128)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    config = NetworkConfig(num_blocks=args.blocks, num_filters=args.filters)
    train(
        data_paths=args.data,
        config=config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
    )
