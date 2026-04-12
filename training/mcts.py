"""Python MCTS implementation for self-play.

Uses python-chess for move generation and PyTorch model for position evaluation.
Batched inference: gathers multiple leaf positions per iteration for a single GPU
forward pass. Virtual loss prevents traversal collisions within a batch.
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
    # Dynamic c_puct: c_init + c_factor * log((N + c_base) / c_base)
    # Dynamic c_puct — Lc0 current defaults
    c_puct_init: float = 3.0
    c_puct_base: float = 19652.0
    c_puct_factor: float = 2.0
    fpu_reduction: float = 1.2
    fpu_reduction_root: float = 1.2
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature: float = 1.0
    policy_softmax_temperature: float = 2.2
    batch_size: int = 16          # leaves gathered before one GPU forward pass
    nn_cache_size: int = 20000    # max cached evaluations (0 = disabled)
    smart_pruning: bool = True    # stop early if best move can't be overtaken
    smart_pruning_factor: float = 1.33  # margin factor for pruning decision
    two_fold_draw: bool = True    # treat 2-fold repetition as draw in search tree
    shaped_dirichlet: bool = True # KataGo-style: concentrate noise on plausible moves
    uncertainty_weight: float = 0.15  # exploration bonus from value variance (0 = disabled)
    variance_scaling: bool = True     # scale c_puct by parent value variance
    contempt: float = 0.0        # draw aversion: positive = prefer wins over draws


@dataclass
class SearchResult:
    best_move: chess.Move | None
    visit_counts: dict[chess.Move, int]
    root_value: float
    policy_target: np.ndarray  # shape (1858,)
    root_node: 'Node | None' = None  # For tree reuse
    raw_policy: np.ndarray | None = None  # NN policy before MCTS (for diff-focus sampling)
    raw_value: float = 0.0  # NN value before MCTS (for diff-focus sampling)


class Node:
    """MCTS tree node."""

    # terminal_status: 0 = unknown, 1 = proven win (for side to move), -1 = proven loss, 2 = proven draw
    __slots__ = ['prior', 'visit_count', 'total_value', 'children', 'terminal_status', 'pending_evals', 'sum_sq_value']

    def __init__(self, prior: float):
        self.prior = prior
        self.visit_count = 0
        self.total_value = 0.0
        self.children: dict[chess.Move, 'Node'] = {}
        self.terminal_status = 0  # unknown
        self.pending_evals = 0
        self.sum_sq_value = 0.0

    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def value_variance(self) -> float:
        if self.visit_count < 2:
            return 0.0
        mean = self.total_value / self.visit_count
        mean_sq = self.sum_sq_value / self.visit_count
        return max(0.0, mean_sq - mean * mean)

    def puct_score(self, parent_visits: int, c_puct: float) -> float:
        """PUCT score using pre-computed dynamic c_puct."""
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.value() + exploration

    def is_expanded(self) -> bool:
        return len(self.children) > 0

    def apply_virtual_loss(self):
        """Increment visit count without value update (discourages re-selection)."""
        self.visit_count += 1
        self.pending_evals += 1

    def revert_virtual_loss(self):
        """Remove virtual loss before real backpropagation."""
        self.visit_count -= 1
        self.pending_evals -= 1


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


class NNCache:
    """Cache for neural network evaluations, keyed by board FEN."""

    def __init__(self, max_size: int = 20000):
        self._cache: dict[str, tuple[np.ndarray, float]] = {}
        self._max_size = max_size

    def get(self, board: chess.Board) -> tuple[np.ndarray, float] | None:
        return self._cache.get(board.fen())

    def put(self, board: chess.Board, policy: np.ndarray, value: float):
        if len(self._cache) >= self._max_size:
            keys = list(self._cache.keys())
            for k in keys[:len(keys) // 4]:
                del self._cache[k]
        self._cache[board.fen()] = (policy, value)

    def clear(self):
        self._cache.clear()

    def __len__(self):
        return len(self._cache)


class MCTS:
    """Monte Carlo Tree Search using a neural network for evaluation."""

    def __init__(self, model: torch.nn.Module, config: MCTSConfig, device: str = 'cpu'):
        self.model = model
        self.config = config
        self.device = device
        self.nn_cache = NNCache(config.nn_cache_size) if config.nn_cache_size > 0 else None

    def search(self, board: chess.Board, root: 'Node | None' = None) -> SearchResult:
        """Run MCTS search from the given board position.

        Args:
            board: Current board position.
            root: Optional pre-existing root node for tree reuse. If provided,
                  skips initial expansion and Dirichlet noise (already applied).
        """
        if board.is_game_over():
            return SearchResult(
                best_move=None,
                visit_counts={},
                root_value=self._terminal_value(board),
                policy_target=np.zeros(POLICY_SIZE, dtype=np.float32),
            )

        if root is not None and root.is_expanded():
            # Tree reuse: skip expansion and noise
            raw_policy = None
            raw_value = root.value()
        else:
            root = Node(prior=1.0)
            raw_policy, raw_value = self._evaluate(board)
            self._expand(root, board, raw_policy)
            self._add_dirichlet_noise(root)
            root.visit_count = 1
            root.total_value = raw_value

        sims_done = 0
        batch_size = self.config.batch_size

        while sims_done < self.config.num_simulations:
            # Smart pruning: stop if best move can't be overtaken
            if self.config.smart_pruning and root.children and sims_done > batch_size:
                remaining = self.config.num_simulations - sims_done
                visits = sorted(
                    [c.visit_count for c in root.children.values()],
                    reverse=True,
                )
                if len(visits) >= 2:
                    best, second = visits[0], visits[1]
                    if second + remaining < best * self.config.smart_pruning_factor:
                        break

            n = min(batch_size, self.config.num_simulations - sims_done)
            pending = []  # (path, scratch_board, leaf_node)

            # === GATHER PHASE ===
            for _ in range(n):
                node = root
                scratch_board = board.copy()
                path = [node]

                while node.is_expanded() and not scratch_board.is_game_over():
                    move, node = self._select_child(node, is_root=(node is root))
                    scratch_board.push(move)
                    path.append(node)
                    # Two-fold repetition → break out as terminal draw
                    if self.config.two_fold_draw and scratch_board.is_repetition(2):
                        break

                # Terminal — resolve immediately
                is_terminal = scratch_board.is_game_over()
                is_rep_draw = (not is_terminal and self.config.two_fold_draw
                               and scratch_board.is_repetition(2))
                if is_terminal or is_rep_draw:
                    leaf_value = self._terminal_value(scratch_board)
                    if scratch_board.is_checkmate():
                        node.terminal_status = 1
                    else:
                        node.terminal_status = 2
                    self._backpropagate(path, leaf_value)
                    self._propagate_terminal(path)
                    sims_done += 1
                    continue

                # Already expanded (another traversal in this batch expanded it)
                if node.is_expanded():
                    self._backpropagate(path, node.value())
                    sims_done += 1
                    continue

                # Check NN cache
                if self.nn_cache is not None:
                    cached = self.nn_cache.get(scratch_board)
                    if cached is not None:
                        policy, value = cached
                        self._expand(node, scratch_board, policy)
                        self._backpropagate(path, value)
                        self._propagate_terminal(path)
                        sims_done += 1
                        continue

                # Needs NN evaluation — apply virtual loss and queue
                for p_node in path:
                    p_node.apply_virtual_loss()
                pending.append((path, scratch_board, node))
                sims_done += 1

            if not pending:
                continue

            # === BATCH EVALUATE ===
            boards_to_eval = [item[1] for item in pending]
            results = self._batch_evaluate(boards_to_eval)

            # === SCATTER PHASE ===
            for (path, scratch_board, leaf_node), (policy, value) in zip(pending, results):
                for p_node in path:
                    p_node.revert_virtual_loss()
                self._expand(leaf_node, scratch_board, policy)
                self._backpropagate(path, value)
                self._propagate_terminal(path)
                if self.nn_cache is not None:
                    self.nn_cache.put(scratch_board, policy, value)

        return self._build_result(root, board, raw_policy, raw_value)

    def reuse_tree(self, root: 'Node', move: chess.Move) -> 'Node | None':
        """Extract subtree for the given move for tree reuse.

        Returns the child node (new root) if it exists and is expanded, else None.
        Not used during self-play training (Lc0 convention: independent root distributions).
        """
        if move in root.children:
            child = root.children[move]
            if child.is_expanded():
                return child
        return None

    def _evaluate(self, board: chess.Board) -> tuple[np.ndarray, float]:
        planes = encode_board(board)
        tensor = torch.from_numpy(planes).unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy_logits, value_logits, _mlh = self.model(tensor)

        # Apply policy softmax temperature (>1.0 widens the distribution)
        pst = self.config.policy_softmax_temperature
        policy_probs = torch.softmax(policy_logits / pst, dim=1).squeeze(0).cpu().numpy()
        wdl = torch.softmax(value_logits, dim=1).squeeze(0).cpu().numpy()
        value = float(wdl[0] - wdl[2])

        return policy_probs, value

    def _batch_evaluate(self, boards: list[chess.Board]) -> list[tuple[np.ndarray, float]]:
        """Evaluate multiple boards in a single GPU forward pass."""
        planes_list = [encode_board(b) for b in boards]
        batch_tensor = torch.from_numpy(np.stack(planes_list)).to(self.device)

        with torch.no_grad():
            policy_logits, value_logits, _mlh = self.model(batch_tensor)

        pst = self.config.policy_softmax_temperature
        policy_probs = torch.softmax(policy_logits / pst, dim=1).cpu().numpy()
        wdl = torch.softmax(value_logits, dim=1).cpu().numpy()
        values = wdl[:, 0] - wdl[:, 2]

        return [(policy_probs[i], float(values[i])) for i in range(len(boards))]

    def _expand(self, node: Node, board: chess.Board, policy: np.ndarray):
        for move in board.legal_moves:
            idx = chess_move_to_policy_index(move, board.turn)
            prior = policy[idx] if idx is not None else 1e-6
            node.children[move] = Node(prior=prior)

        total_prior = sum(child.prior for child in node.children.values())
        if total_prior > 0:
            for child in node.children.values():
                child.prior /= total_prior

    def _dynamic_cpuct(self, parent_visits: int) -> float:
        """Compute dynamic c_puct that grows with parent visit count."""
        cfg = self.config
        return cfg.c_puct_init + cfg.c_puct_factor * math.log(
            (parent_visits + cfg.c_puct_base) / cfg.c_puct_base
        )

    def _select_child(self, node: Node, is_root: bool = False) -> tuple[chess.Move, 'Node']:
        best_score = -float('inf')
        best_move = None
        best_child = None

        fpu_red = self.config.fpu_reduction_root if is_root else self.config.fpu_reduction
        fpu_value = node.value() - fpu_red
        c_puct = self._dynamic_cpuct(node.visit_count)

        # Variance-scaled cPUCT: scale exploration by parent value variance
        if self.config.variance_scaling and node.visit_count > 1:
            variance_scale = max(0.5, min(2.0, math.sqrt(node.value_variance()) / 0.5))
            c_puct *= variance_scale

        for move, child in node.children.items():
            # MCTS-solver: child.terminal_status is from child's STM (opponent).
            # child=1 (opponent loses) = win for us → pick immediately
            # child=-1 (opponent wins) = loss for us → skip
            if child.terminal_status == 1:
                return move, child
            if child.terminal_status == -1:
                continue

            if child.visit_count == 0:
                score = fpu_value + c_puct * child.prior * math.sqrt(node.visit_count)
            else:
                score = child.puct_score(node.visit_count, c_puct)
                if self.config.uncertainty_weight > 0:
                    score += self.config.uncertainty_weight * math.sqrt(child.value_variance())
            if score > best_score:
                best_score = score
                best_move = move
                best_child = child

        # If all children are proven losses, pick any (we're losing)
        if best_child is None:
            move, child = next(iter(node.children.items()))
            return move, child

        return best_move, best_child

    def _add_dirichlet_noise(self, root: Node):
        if not root.children:
            return

        n = len(root.children)
        eps = self.config.dirichlet_epsilon

        if not self.config.shaped_dirichlet:
            # Original uniform Dirichlet
            noise = np.random.dirichlet([self.config.dirichlet_alpha] * n)
        else:
            # KataGo-style: concentrate noise on plausible moves
            priors = np.array([c.prior for c in root.children.values()])
            log_priors = np.log(priors + 1e-8)
            threshold = log_priors.max() - 2.0  # moves within ~7x of best
            weights = np.ones_like(log_priors) * 0.5
            above = log_priors > threshold
            weights[above] += 0.5 * (log_priors[above] - threshold) / 2.0
            weights /= weights.sum()
            scaled_alpha = self.config.dirichlet_alpha * weights * n
            noise = np.random.dirichlet(np.maximum(scaled_alpha, 0.01))

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
            node.sum_sq_value += value * value
            node.visit_count += 1
            value = -value

    def _propagate_terminal(self, path: list[Node]):
        """Propagate proven terminal results up the tree (MCTS-solver).

        terminal_status convention (always from this node's side-to-move perspective):
            0 = unknown, 1 = proven loss for STM, -1 = proven win for STM, 2 = proven draw

        A child's status is from the child's STM perspective (our opponent).
        So child_status=1 (child loses) means WE win → parent_status=-1.
        """
        for node in reversed(path):
            if node.terminal_status != 0 or not node.children:
                continue

            # If any child is a loss for child's STM → we can win
            for child in node.children.values():
                if child.terminal_status == 1:
                    node.terminal_status = -1  # proven win for us
                    break

            if node.terminal_status != 0:
                continue  # just got proven

            # If any child is still unknown, we can't prove this node
            if any(c.terminal_status == 0 for c in node.children.values()):
                break

            # All children are proven (win or draw for child's STM = loss or draw for us)
            # Best case: if any child is a draw, we can force a draw
            if any(c.terminal_status == 2 for c in node.children.values()):
                node.terminal_status = 2  # proven draw
            else:
                node.terminal_status = 1  # all children win (for opponent) = we lose

    def _build_result(
        self, root: Node, board: chess.Board,
        raw_policy: np.ndarray | None = None, raw_value: float = 0.0,
    ) -> SearchResult:
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

        # Contempt: shift value away from 0 (draws) toward the existing sign
        if self.config.contempt > 0 and abs(root_value) < 1.0:
            sign = 1.0 if root_value >= 0 else -1.0
            shift = self.config.contempt * (1.0 - abs(root_value))
            root_value = max(-1.0, min(1.0, root_value + sign * shift))

        return SearchResult(
            best_move=best_move,
            visit_counts=visit_counts,
            root_value=root_value,
            policy_target=policy_target,
            root_node=root,
            raw_policy=raw_policy,
            raw_value=raw_value,
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
