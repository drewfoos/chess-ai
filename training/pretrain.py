"""Supervised pretraining loop (Phase A / Phase B).

Trains a ChessNetwork on sharded .npz files produced by `pretrain_dataset.py`
(phase A, one-hot moves + WDL) or by `stockfish_label.py` (phase B, soft policy
+ cp-derived WDL).

Streaming design: loads a small group of shards into RAM, trains over them, then
moves to the next group. Keeps peak memory bounded regardless of total corpus
size. After the full pretraining run, writes a checkpoint whose format matches
`training_loop --resume-from` so the self-play RL loop can continue from it.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR, MultiStepLR, SequentialLR
from torch.utils.data import DataLoader, IterableDataset

from training.config import NetworkConfig
from training.dataset import ChessDataset
from training.model import ChessNetwork
from training.train import compute_loss, create_optimizer


class StreamingShardDataset(IterableDataset):
    """Streams positions through one persistent DataLoader for the whole run.

    Why: the original loop created a fresh DataLoader per shard group, which on
    Windows leaked kernel shared-memory file mappings (OSError 1450, "no system
    resources") after ~50 iterations with num_workers>0. A single persistent
    loader backed by this iterable avoids repeated worker spawning.

    Memory stays bounded because each worker only holds `shards_per_group`
    shards in RAM at a time and drops them before loading the next group.
    """

    def __init__(
        self,
        shard_paths: list[str],
        shards_per_group: int,
        epochs: int,
        shuffle: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.shard_paths = list(shard_paths)
        self.shards_per_group = shards_per_group
        self.epochs = epochs
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid, nw = (0, 1) if info is None else (info.id, info.num_workers)

        for epoch in range(self.epochs):
            rng = random.Random(self.seed + epoch * 10007 + wid)
            paths = list(self.shard_paths)
            if self.shuffle:
                rng.shuffle(paths)
            # Round-robin shard partition: each worker handles a disjoint slice.
            paths = paths[wid::nw]

            for i in range(0, len(paths), self.shards_per_group):
                group = paths[i:i + self.shards_per_group]
                ds = ChessDataset(group)
                order = list(range(len(ds)))
                if self.shuffle:
                    rng.shuffle(order)
                for j in order:
                    yield ds[j]
                del ds


def pretrain(
    shard_dir: str,
    out_checkpoint: str,
    blocks: int = 14,
    filters: int = 192,
    epochs: int = 1,
    batch_size: int = 1024,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    warmup_steps: int = 1000,
    grad_clip: float = 1.0,
    lr_milestones: list[int] | None = None,
    lr_gamma: float = 0.1,
    shards_per_group: int = 4,
    device: str = 'auto',
    num_workers: int = 2,
    resume_from: str | None = None,
    soft_policy_weight: float = 0.0,
    log_every: int = 50,
    checkpoint_every: int = 2000,
):
    """Run supervised pretraining over every `shard_*.npz` under `shard_dir`."""
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # BF16 AMP (not FP16): the attention policy head can produce logits
    # exceeding FP16's 65504 ceiling mid-training, which turns log_softmax
    # into NaN. BF16 has FP32's exponent range, same tensor-core speed on
    # Ampere (RTX 3080), and doesn't need GradScaler.
    use_amp = (device == 'cuda')
    amp_dtype = torch.bfloat16

    # Recursive so Phase B's orchestrator layout (part_NN/shard_*.npz) works
    # alongside Phase A's flat layout.
    shard_paths = sorted(
        glob.glob(os.path.join(shard_dir, 'shard_*.npz'))
        + glob.glob(os.path.join(shard_dir, '**', 'shard_*.npz'), recursive=True)
    )
    shard_paths = sorted(set(shard_paths))
    if not shard_paths:
        raise FileNotFoundError(f"No shard_*.npz files under {shard_dir}")
    print(f"Found {len(shard_paths)} shards under {shard_dir}")

    # Model + optimizer. Resume-from lets phase B continue from phase A weights.
    config = NetworkConfig(num_blocks=blocks, num_filters=filters)
    model = ChessNetwork(config).to(device)
    if resume_from:
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        saved_cfg = ckpt.get('config')
        if saved_cfg is not None and (
            saved_cfg.num_blocks != config.num_blocks
            or saved_cfg.num_filters != config.num_filters
        ):
            print(
                f"  [resume] rebuilding model with saved config "
                f"({saved_cfg.num_blocks}b×{saved_cfg.num_filters}f)"
            )
            config = saved_cfg
            model = ChessNetwork(config).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Resumed weights from {resume_from}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {config.num_blocks}b×{config.num_filters}f ({total_params:,} params)")

    optimizer = create_optimizer(model, lr=lr, weight_decay=weight_decay)

    warmup = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps)
    if lr_milestones:
        decay = MultiStepLR(optimizer, milestones=lr_milestones, gamma=lr_gamma)
        scheduler = SequentialLR(optimizer, [warmup, decay], milestones=[warmup_steps])
        print(f"LR: warmup {warmup_steps} steps, decay at {lr_milestones} (gamma={lr_gamma})")
    else:
        scheduler = warmup
        print(f"LR: warmup {warmup_steps} steps (no step decay)")

    os.makedirs(os.path.dirname(out_checkpoint) or '.', exist_ok=True)

    streaming = StreamingShardDataset(
        shard_paths=shard_paths,
        shards_per_group=shards_per_group,
        epochs=epochs,
        shuffle=True,
        seed=42,
    )
    loader = DataLoader(
        streaming,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device == 'cuda'),
        persistent_workers=(num_workers > 0),
        drop_last=True,
    )
    print(
        f"Streaming {len(shard_paths)} shards × {epochs} epochs, "
        f"{shards_per_group}/group, {num_workers} workers, batch {batch_size}"
    )

    step = 0
    t_start = time.time()
    last_log_time = t_start
    last_log_step = 0
    running_loss = 0.0
    running_policy = 0.0
    running_value = 0.0
    running_count = 0

    def _save_checkpoint() -> None:
        torch.save({
            'generation': 0,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
            'step': step,
            'pretrain': True,
        }, out_checkpoint)

    for batch in loader:
        planes = batch[0].to(device, non_blocking=True)
        policies = batch[1].to(device, non_blocking=True)
        values = batch[2].to(device, non_blocking=True)
        mlh_target = batch[3].to(device, non_blocking=True) if len(batch) > 3 else None

        model.train()
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
            policy_logits, value_logits, mlh_pred = model(planes)
            loss, p_loss, v_loss, _sp, _mlh = compute_loss(
                policy_logits, value_logits, policies, values,
                mlh_pred, mlh_target,
                soft_policy_weight=soft_policy_weight,
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        scheduler.step()

        step += 1
        running_loss += loss.item()
        running_policy += p_loss.item()
        running_value += v_loss.item()
        running_count += 1

        if step % log_every == 0:
            now = time.time()
            lr_now = optimizer.param_groups[0]['lr']
            pos_per_sec = (step - last_log_step) * batch_size / max(now - last_log_time, 1e-6)
            avg_loss = running_loss / running_count
            avg_p = running_policy / running_count
            avg_v = running_value / running_count
            print(
                f"  step {step:>7} | "
                f"loss {avg_loss:.4f} p {avg_p:.4f} v {avg_v:.4f} | "
                f"lr {lr_now:.2e} | {pos_per_sec:,.0f} pos/s"
            )
            last_log_time = now
            last_log_step = step
            running_loss = running_policy = running_value = 0.0
            running_count = 0

        if step % checkpoint_every == 0:
            _save_checkpoint()

    torch.save({
        'generation': 0,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'step': step,
        'pretrain': True,
    }, out_checkpoint)
    print(f"\nFinal checkpoint saved to {out_checkpoint} ({step} steps total)")


def main():
    parser = argparse.ArgumentParser(description='Supervised pretraining on .npz shards')
    parser.add_argument('--shard-dir', required=True, help='Directory of shard_*.npz files')
    parser.add_argument('--out', required=True, help='Output checkpoint path (.pt)')
    parser.add_argument('--blocks', type=int, default=20)
    parser.add_argument('--filters', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--warmup-steps', type=int, default=1000)
    parser.add_argument('--grad-clip', type=float, default=1.0,
                        help='Max gradient norm. Lower = more stable under AMP/FP16, '
                             'but slower to converge. 1.0 is the standard AdamW default; '
                             'raise to 5.0-10.0 if you see the grad scaler warning a lot.')
    parser.add_argument('--lr-milestones', type=str, default='',
                        help='Comma-separated step numbers for LR decay, e.g. "50000,100000"')
    parser.add_argument('--lr-gamma', type=float, default=0.1)
    parser.add_argument('--shards-per-group', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--resume-from', default=None,
                        help='Path to a previous checkpoint to continue from (e.g. phase A result for phase B)')
    parser.add_argument('--soft-policy-weight', type=float, default=0.0,
                        help='Weight for KataGo-style aux soft-policy loss. '
                             '0.0 for phase A (one-hot targets), 0.2 for phase B if you have soft targets.')
    parser.add_argument('--log-every', type=int, default=50)
    parser.add_argument('--checkpoint-every', type=int, default=2000,
                        help='Save a rolling checkpoint every N training steps (default 2000)')
    args = parser.parse_args()

    milestones = [int(m) for m in args.lr_milestones.split(',') if m.strip()] or None

    pretrain(
        shard_dir=args.shard_dir,
        out_checkpoint=args.out,
        blocks=args.blocks,
        filters=args.filters,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        lr_milestones=milestones,
        lr_gamma=args.lr_gamma,
        shards_per_group=args.shards_per_group,
        device=args.device,
        num_workers=args.num_workers,
        resume_from=args.resume_from,
        soft_policy_weight=args.soft_policy_weight,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
    )


if __name__ == '__main__':
    main()
