"""Python MCTS implementation for self-play.

Uses python-chess for move generation and PyTorch model for position evaluation.
Single-threaded, single-position inference — correct and simple, not fast.
"""

import math
from dataclasses import dataclass, field

import chess
import numpy as np
import torch

from training.config import NetworkConfig
from training.encoder import encode_board, move_to_index, mirror_move, POLICY_SIZE


@dataclass
class MCTSConfig:
    num_simulations: int = 400
    c_puct: float = 2.5
    fpu_reduction: float = 0.25
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature: float = 1.0


@dataclass
class SearchResult:
    best_move: chess.Move | None
    visit_counts: dict[chess.Move, int]
    root_value: float
    policy_target: np.ndarray  # shape (1858,)


class Node:
    """MCTS tree node."""

    __slots__ = ['prior', 'visit_count', 'total_value', 'children']

    def __init__(self, prior: float):
        self.prior = prior
        self.visit_count = 0
        self.total_value = 0.0
        self.children: dict[chess.Move, 'Node'] = {}

    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def puct_score(self, parent_visits: int, c_puct: float) -> float:
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.value() + exploration

    def is_expanded(self) -> bool:
        return len(self.children) > 0


def chess_move_to_policy_index(move: chess.Move, turn: chess.Color) -> int | None:
    """Convert a python-chess Move to a policy index in [0, 1857]."""
    from_sq = move.from_square
    to_sq = move.to_square

    if turn == chess.BLACK:
        from_sq = mirror_move(from_sq)
        to_sq = mirror_move(to_sq)

    promo = None
    if move.promotion is not None and move.promotion != chess.QUEEN:
        promo_map = {chess.KNIGHT: 'n', chess.BISHOP: 'b', chess.ROOK: 'r'}
        promo = promo_map.get(move.promotion)

    return move_to_index(from_sq, to_sq, promo)


class MCTS:
    """Monte Carlo Tree Search using a neural network for evaluation."""

    def __init__(self, model: torch.nn.Module, config: MCTSConfig, device: str = 'cpu'):
        self.model = model
        self.config = config
        self.device = device

    def search(self, board: chess.Board) -> SearchResult:
        if board.is_game_over():
            return SearchResult(
                best_move=None,
                visit_counts={},
                root_value=self._terminal_value(board),
                policy_target=np.zeros(POLICY_SIZE, dtype=np.float32),
            )

        root = Node(prior=1.0)
        policy, value = self._evaluate(board)
        self._expand(root, board, policy)
        self._add_dirichlet_noise(root)

        root.visit_count = 1
        root.total_value = value

        for _ in range(self.config.num_simulations):
            node = root
            scratch_board = board.copy()
            path = [node]

            while node.is_expanded() and not scratch_board.is_game_over():
                move, node = self._select_child(node)
                scratch_board.push(move)
                path.append(node)

            if scratch_board.is_game_over():
                leaf_value = self._terminal_value(scratch_board)
            elif not node.is_expanded():
                policy, leaf_value = self._evaluate(scratch_board)
                self._expand(node, scratch_board, policy)
            else:
                leaf_value = self._terminal_value(scratch_board)

            self._backpropagate(path, leaf_value)

        return self._build_result(root, board)

    def _evaluate(self, board: chess.Board) -> tuple[np.ndarray, float]:
        planes = encode_board(board)
        tensor = torch.from_numpy(planes).unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy_logits, value_logits = self.model(tensor)

        policy_probs = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
        wdl = torch.softmax(value_logits, dim=1).squeeze(0).cpu().numpy()
        value = float(wdl[0] - wdl[2])

        return policy_probs, value

    def _expand(self, node: Node, board: chess.Board, policy: np.ndarray):
        for move in board.legal_moves:
            idx = chess_move_to_policy_index(move, board.turn)
            prior = policy[idx] if idx is not None else 1e-6
            node.children[move] = Node(prior=prior)

        total_prior = sum(child.prior for child in node.children.values())
        if total_prior > 0:
            for child in node.children.values():
                child.prior /= total_prior

    def _select_child(self, node: Node) -> tuple[chess.Move, 'Node']:
        best_score = -float('inf')
        best_move = None
        best_child = None

        fpu_value = node.value() - self.config.fpu_reduction

        for move, child in node.children.items():
            if child.visit_count == 0:
                score = fpu_value + self.config.c_puct * child.prior * math.sqrt(node.visit_count)
            else:
                score = child.puct_score(node.visit_count, self.config.c_puct)
            if score > best_score:
                best_score = score
                best_move = move
                best_child = child

        return best_move, best_child

    def _add_dirichlet_noise(self, root: Node):
        if not root.children:
            return
        noise = np.random.dirichlet([self.config.dirichlet_alpha] * len(root.children))
        eps = self.config.dirichlet_epsilon
        for i, child in enumerate(root.children.values()):
            child.prior = (1 - eps) * child.prior + eps * noise[i]

    def _terminal_value(self, board: chess.Board) -> float:
        if board.is_checkmate():
            return -1.0
        return 0.0

    def _backpropagate(self, path: list[Node], leaf_value: float):
        value = leaf_value
        for node in reversed(path):
            node.total_value += value
            node.visit_count += 1
            value = -value

    def _build_result(self, root: Node, board: chess.Board) -> SearchResult:
        visit_counts = {move: child.visit_count for move, child in root.children.items()}

        best_move = self._select_move(root, board)

        policy_target = np.zeros(POLICY_SIZE, dtype=np.float32)
        total_visits = sum(visit_counts.values())
        if total_visits > 0:
            for move, visits in visit_counts.items():
                idx = chess_move_to_policy_index(move, board.turn)
                if idx is not None:
                    policy_target[idx] = visits / total_visits

        root_value = root.value()
        return SearchResult(
            best_move=best_move,
            visit_counts=visit_counts,
            root_value=root_value,
            policy_target=policy_target,
        )

    def _select_move(self, root: Node, board: chess.Board) -> chess.Move:
        moves = list(root.children.keys())
        visits = np.array([root.children[m].visit_count for m in moves], dtype=np.float64)

        if self.config.temperature <= 0.01:
            best_idx = np.argmax(visits)
        else:
            visits = visits ** (1.0 / self.config.temperature)
            probs = visits / visits.sum()
            best_idx = np.random.choice(len(moves), p=probs)

        return moves[best_idx]
