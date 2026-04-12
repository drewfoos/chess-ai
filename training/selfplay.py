"""Self-play game generation and reinforcement learning loop.

Usage:
    python -m training.selfplay generate --games 50 --simulations 100
    python -m training.selfplay loop --generations 10 --games-per-gen 50
"""

import argparse
import os
import time
from dataclasses import dataclass, field

import chess
import numpy as np
import torch

from training.config import NetworkConfig
from training.encoder import encode_board, POLICY_SIZE
from training.model import ChessNetwork
from training.mcts import MCTS, MCTSConfig


@dataclass
class SelfPlayConfig:
    temperature_moves: int = 30
    max_moves: int = 512
    resign_threshold: float = -0.95
    consecutive_resign: int = 5


@dataclass
class GameRecord:
    planes: list[np.ndarray] = field(default_factory=list)
    policies: list[np.ndarray] = field(default_factory=list)
    values: list[np.ndarray] = field(default_factory=list)
    result: str = '*'
    num_moves: int = 0
    moves_uci: list[str] = field(default_factory=list)


def play_game(mcts: MCTS, config: SelfPlayConfig) -> GameRecord:
    """Play a single self-play game."""
    board = chess.Board()
    record = GameRecord()

    positions = []  # (planes, policy_target, side_to_move)
    resign_count = 0

    for move_num in range(config.max_moves):
        if board.is_game_over(claim_draw=True):
            break

        # Set temperature based on move number
        if move_num < config.temperature_moves:
            mcts.config.temperature = 1.0
        else:
            mcts.config.temperature = 0.01

        result = mcts.search(board)
        if result.best_move is None:
            break

        planes = encode_board(board)
        positions.append((planes, result.policy_target, board.turn))

        # Check resign condition
        if result.root_value < config.resign_threshold:
            resign_count += 1
            if resign_count >= config.consecutive_resign:
                if board.turn == chess.WHITE:
                    record.result = '0-1'
                else:
                    record.result = '1-0'
                break
        else:
            resign_count = 0

        record.moves_uci.append(result.best_move.uci())
        board.push(result.best_move)

    # Determine game result if not resigned
    if record.result == '*':
        outcome = board.outcome(claim_draw=True)
        if outcome is None:
            record.result = '1/2-1/2'
        elif outcome.winner is None:
            record.result = '1/2-1/2'
        elif outcome.winner == chess.WHITE:
            record.result = '1-0'
        else:
            record.result = '0-1'

    # Label all positions with WDL
    for planes, policy_target, side_to_move in positions:
        if record.result == '1/2-1/2':
            wdl = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        elif record.result == '1-0':
            if side_to_move == chess.WHITE:
                wdl = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            else:
                wdl = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:  # 0-1
            if side_to_move == chess.BLACK:
                wdl = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            else:
                wdl = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        record.planes.append(planes)
        record.policies.append(policy_target)
        record.values.append(wdl)

    record.num_moves = len(positions)
    return record


def generate_games(
    model: ChessNetwork,
    num_games: int,
    output_path: str,
    mcts_config: MCTSConfig = MCTSConfig(),
    selfplay_config: SelfPlayConfig = SelfPlayConfig(),
    device: str = 'cpu',
    metrics_logger=None,
) -> int:
    """Generate self-play games and save as .npz file."""
    model = model.to(device)
    model.eval()
    mcts = MCTS(model, mcts_config, device=device)

    all_planes = []
    all_policies = []
    all_values = []

    start = time.time()

    for game_num in range(num_games):
        game_start = time.time()
        record = play_game(mcts, selfplay_config)
        game_time = time.time() - game_start

        if metrics_logger is not None:
            from training.metrics import GameMetrics
            metrics_logger.record_game(GameMetrics(
                game_num=game_num + 1,
                num_moves=record.num_moves,
                result=record.result,
                duration_s=game_time,
                moves_uci=record.moves_uci,
            ))

        all_planes.extend(record.planes)
        all_policies.extend(record.policies)
        all_values.extend(record.values)

        total_positions = len(all_planes)
        print(
            f"  Game {game_num + 1}/{num_games}: "
            f"{record.num_moves} moves, {record.result}, "
            f"{game_time:.1f}s ({total_positions} positions total)"
        )

    elapsed = time.time() - start
    total_positions = len(all_planes)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    np.savez(
        output_path,
        planes=np.array(all_planes, dtype=np.float32),
        policies=np.array(all_policies, dtype=np.float32),
        values=np.array(all_values, dtype=np.float32),
    )

    print(f"Generated {num_games} games, {total_positions} positions in {elapsed:.1f}s")
    print(f"Saved to {output_path}")
    return total_positions


def training_loop(
    generations: int = 10,
    games_per_gen: int = 50,
    train_epochs: int = 5,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    num_simulations: int = 400,
    blocks: int = 10,
    filters: int = 128,
    window_size: int = 5,
    output_dir: str = 'selfplay_output',
    device: str = 'auto',
    max_moves: int = 512,
    resign_threshold: float = -0.95,
):
    """Full reinforcement learning training loop.

    For each generation:
      1. Generate self-play games and save as data/gen_NNN.npz
      2. Load sliding window of recent generations into ChessDataset
      3. Train for train_epochs epochs
      4. Save checkpoint as checkpoints/model_gen_N.pt
      5. Print per-generation stats

    After all generations, export final TorchScript model.
    """
    from torch.utils.data import DataLoader
    from training.dataset import ChessDataset
    from training.train import compute_loss, create_optimizer
    from training.export import export_torchscript

    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_dir = os.path.join(output_dir, 'data')
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    from training.metrics import MetricsLogger, TrainingMetrics
    metrics_dir = os.path.join(output_dir, 'metrics')
    metrics_logger = MetricsLogger(metrics_dir)

    config = NetworkConfig(num_blocks=blocks, num_filters=filters)
    model = ChessNetwork(config).to(device)
    optimizer = create_optimizer(model, lr=lr, weight_decay=weight_decay)

    mcts_config = MCTSConfig(num_simulations=num_simulations)
    selfplay_config = SelfPlayConfig(max_moves=max_moves, resign_threshold=resign_threshold)

    print(f"Starting training loop: {generations} generations, {games_per_gen} games/gen")
    print(f"Device: {device}, Model: {blocks} blocks, {filters} filters")

    for gen in range(1, generations + 1):
        gen_start = time.time()
        print(f"\n{'='*60}")
        print(f"Generation {gen}/{generations}")
        print(f"{'='*60}")

        # 1. Generate self-play games
        data_path = os.path.join(data_dir, f'gen_{gen:03d}.npz')
        model.eval()
        num_positions = generate_games(
            model, games_per_gen, data_path,
            mcts_config=mcts_config,
            selfplay_config=selfplay_config,
            device=device,
            metrics_logger=metrics_logger,
        )

        # 2. Load sliding window of recent generations
        window_start = max(1, gen - window_size + 1)
        npz_paths = [
            os.path.join(data_dir, f'gen_{g:03d}.npz')
            for g in range(window_start, gen + 1)
        ]
        dataset = ChessDataset(npz_paths)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        # 3. Train for train_epochs epochs
        model.train()
        total_loss_sum = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        num_batches = 0

        for epoch in range(train_epochs):
            for planes, policy_target, value_target in dataloader:
                planes = planes.to(device)
                policy_target = policy_target.to(device)
                value_target = value_target.to(device)

                optimizer.zero_grad()
                policy_logits, value_logits = model(planes)
                total_loss, policy_loss, value_loss = compute_loss(
                    policy_logits, value_logits, policy_target, value_target
                )
                total_loss.backward()
                optimizer.step()

                total_loss_sum += total_loss.item()
                policy_loss_sum += policy_loss.item()
                value_loss_sum += value_loss.item()
                num_batches += 1

        # 4. Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'model_gen_{gen}.pt')
        torch.save({
            'generation': gen,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
        }, checkpoint_path)

        # 5. Print per-generation stats
        gen_time = time.time() - gen_start
        avg_total = total_loss_sum / max(num_batches, 1)
        avg_policy = policy_loss_sum / max(num_batches, 1)
        avg_value = value_loss_sum / max(num_batches, 1)
        print(f"Gen {gen} complete: {num_positions} positions, "
              f"{len(dataset)} training samples (window {window_start}-{gen})")
        print(f"  Loss: total={avg_total:.4f}, policy={avg_policy:.4f}, value={avg_value:.4f}")
        print(f"  Time: {gen_time:.1f}s, Checkpoint: {checkpoint_path}")
        training_metrics = TrainingMetrics(
            total_loss=avg_total,
            policy_loss=avg_policy,
            value_loss=avg_value,
            num_batches=num_batches,
            learning_rate=optimizer.param_groups[0]['lr'],
        )
        metrics_logger.save_generation(
            generation=gen,
            num_positions=num_positions,
            training=training_metrics,
            duration_s=gen_time,
        )

    # Export final TorchScript model
    final_path = os.path.join(output_dir, 'model_final.pt')
    export_torchscript(model, final_path, device=device)
    print(f"\nTraining complete. Final model exported to {final_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Self-play training for chess AI')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Generate subcommand
    gen_parser = subparsers.add_parser('generate', help='Generate self-play games')
    gen_parser.add_argument('--games', type=int, default=50)
    gen_parser.add_argument('--simulations', type=int, default=400)
    gen_parser.add_argument('--output', type=str, default='data/selfplay.npz')
    gen_parser.add_argument('--blocks', type=int, default=10)
    gen_parser.add_argument('--filters', type=int, default=128)
    gen_parser.add_argument('--device', type=str, default='auto')
    gen_parser.add_argument('--max-moves', type=int, default=512)

    # Loop subcommand
    loop_parser = subparsers.add_parser('loop', help='Run full RL training loop')
    loop_parser.add_argument('--generations', type=int, default=10)
    loop_parser.add_argument('--games-per-gen', type=int, default=50)
    loop_parser.add_argument('--train-epochs', type=int, default=5)
    loop_parser.add_argument('--batch-size', type=int, default=256)
    loop_parser.add_argument('--lr', type=float, default=1e-3)
    loop_parser.add_argument('--simulations', type=int, default=400)
    loop_parser.add_argument('--blocks', type=int, default=10)
    loop_parser.add_argument('--filters', type=int, default=128)
    loop_parser.add_argument('--window-size', type=int, default=5)
    loop_parser.add_argument('--output-dir', type=str, default='selfplay_output')
    loop_parser.add_argument('--device', type=str, default='auto')
    loop_parser.add_argument('--max-moves', type=int, default=512)

    args = parser.parse_args()

    if args.command == 'generate':
        device = args.device
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        config = NetworkConfig(num_blocks=args.blocks, num_filters=args.filters)
        model = ChessNetwork(config).to(device)
        model.eval()

        mcts_config = MCTSConfig(num_simulations=args.simulations)
        selfplay_config = SelfPlayConfig(max_moves=args.max_moves)

        generate_games(
            model, args.games, args.output,
            mcts_config=mcts_config,
            selfplay_config=selfplay_config,
            device=device,
        )

    elif args.command == 'loop':
        training_loop(
            generations=args.generations,
            games_per_gen=args.games_per_gen,
            train_epochs=args.train_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            num_simulations=args.simulations,
            blocks=args.blocks,
            filters=args.filters,
            window_size=args.window_size,
            output_dir=args.output_dir,
            device=args.device,
            max_moves=args.max_moves,
        )
