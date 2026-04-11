"""Self-play game generation and reinforcement learning loop.

Usage:
    python -m training.selfplay generate --games 50 --simulations 100
    python -m training.selfplay loop --generations 10 --games-per-gen 50
"""

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
