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
from torch.optim.lr_scheduler import MultiStepLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from training.config import NetworkConfig
from training.model import ChessNetwork
from training.dataset import ChessDataset


def compute_loss(
    policy_logits: torch.Tensor,
    value_logits: torch.Tensor,
    policy_target: torch.Tensor,
    value_target: torch.Tensor,
    mlh_pred: torch.Tensor | None = None,
    mlh_target: torch.Tensor | None = None,
    policy_mask: torch.Tensor | None = None,
    soft_policy_weight: float = 2.0,
    soft_policy_temperature: float = 4.0,
    wdl_label_smoothing: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute combined policy + value + moves-left + soft policy loss.

    Args:
        policy_logits: Raw logits from policy head (B, 1858)
        value_logits: Raw logits from value head (B, 3)
        policy_target: MCTS visit distribution target (B, 1858)
        value_target: WDL target (B, 3)
        mlh_pred: Moves-left prediction from MLH head (B,), optional
        mlh_target: Moves-left target (B,), optional
        policy_mask: Boolean mask (B,) — True = use policy loss for this sample (playout cap)
        soft_policy_weight: Weight for auxiliary soft policy loss (KataGo-style)
        soft_policy_temperature: Temperature for soft policy target (higher = softer)

    Returns:
        (total_loss, policy_loss, value_loss) tuple.
    """
    # Policy loss: cross-entropy with soft targets
    log_probs = F.log_softmax(policy_logits, dim=1)
    per_sample_policy = -(policy_target * log_probs).sum(dim=1)
    if policy_mask is not None and policy_mask.any() and not policy_mask.all():
        # Only compute policy loss for full-search positions
        policy_loss = per_sample_policy[policy_mask].mean()
    else:
        policy_loss = per_sample_policy.mean()

    # Auxiliary soft policy loss (KataGo): raise search distribution to power 1/T,
    # renormalize, and compute cross-entropy. Forces network to learn about non-obvious moves.
    if soft_policy_weight > 0:
        soft_target = policy_target.pow(1.0 / soft_policy_temperature)
        soft_target = soft_target / (soft_target.sum(dim=1, keepdim=True) + 1e-8)
        per_sample_soft = -(soft_target * log_probs).sum(dim=1)
        if policy_mask is not None and policy_mask.any() and not policy_mask.all():
            soft_policy_loss = per_sample_soft[policy_mask].mean()
        else:
            soft_policy_loss = per_sample_soft.mean()
        policy_loss = policy_loss + soft_policy_weight * soft_policy_loss

    # Value loss: cross-entropy with soft WDL targets.
    # Label smoothing (Lc0/KataGo): blend hard target with uniform [1/3,1/3,1/3]
    # to prevent value-head overconfidence on drawn/near-drawn positions.
    if wdl_label_smoothing > 0:
        smoothed_target = (1.0 - wdl_label_smoothing) * value_target + (wdl_label_smoothing / 3.0)
    else:
        smoothed_target = value_target
    value_log_probs = F.log_softmax(value_logits, dim=1)
    value_loss = -(smoothed_target * value_log_probs).sum(dim=1).mean()

    total = policy_loss + value_loss

    # Moves-left loss: Huber loss (delta=10), scaled by 1/20 (Lc0 convention)
    if mlh_pred is not None and mlh_target is not None:
        mlh_loss = F.huber_loss(mlh_pred, mlh_target, delta=10.0) / 20.0
        total = total + mlh_loss

    return total, policy_loss, value_loss


def create_optimizer(
    model: ChessNetwork,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    optimizer_type: str = 'adamw',
):
    """Create optimizer. Supports 'adamw' (default) and 'sgd' (Lc0-style Nesterov)."""
    if optimizer_type == 'sgd':
        return torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9,
            nesterov=True, weight_decay=weight_decay,
        )
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

    policy_logits, value_logits, _ = model(planes)
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

    # LR warmup (250 steps) + optional step decay
    warmup_steps = 250
    warmup_scheduler = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps)
    if lr_milestones:
        decay_scheduler = MultiStepLR(optimizer, milestones=lr_milestones, gamma=lr_gamma)
        scheduler = SequentialLR(optimizer, [warmup_scheduler, decay_scheduler], milestones=[warmup_steps])
        print(f"LR schedule: warmup {warmup_steps} steps, milestones={lr_milestones}, gamma={lr_gamma}")
    else:
        scheduler = warmup_scheduler
        print(f"LR schedule: warmup {warmup_steps} steps")

    os.makedirs(checkpoint_dir, exist_ok=True)

    use_amp = (device == 'cuda')
    scaler = torch.amp.GradScaler(enabled=use_amp)

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        num_batches = 0
        start = time.time()

        for batch in loader:
            planes = batch[0].to(device)
            policies = batch[1].to(device)
            values = batch[2].to(device)
            mlh_target = batch[3].to(device) if len(batch) > 3 else None

            model.train()
            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device, enabled=use_amp):
                policy_logits, value_logits, mlh_pred = model(planes)
                loss, p_loss, v_loss = compute_loss(
                    policy_logits, value_logits, policies, values,
                    mlh_pred, mlh_target,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()

            scheduler.step()

            epoch_loss += loss.item()
            epoch_policy_loss += p_loss.item()
            epoch_value_loss += v_loss.item()
            num_batches += 1

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
