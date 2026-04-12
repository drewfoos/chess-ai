"""Self-play game generation and reinforcement learning loop.

Usage:
    python -m training.selfplay generate --games 50 --simulations 100
    python -m training.selfplay loop --generations 10 --games-per-gen 50
"""

import argparse
import os
import random
import time
from dataclasses import dataclass, field

import chess
import numpy as np
import torch

from training.config import NetworkConfig
from training.encoder import encode_board, POLICY_SIZE
from training.model import ChessNetwork
from training.mcts import MCTS, MCTSConfig, SearchResult

# Try to import C++ MCTS bindings
try:
    import chess_mcts
    HAS_CPP_MCTS = True
except ImportError:
    HAS_CPP_MCTS = False


class _CppMCTSConfig:
    """Mutable config proxy so play_game can adjust temperature/sims."""

    def __init__(self, mcts_config: MCTSConfig):
        self.temperature = mcts_config.temperature
        self.num_simulations = mcts_config.num_simulations
        self._base = mcts_config

    def _to_dict(self) -> dict:
        """Build config dict for C++ SearchEngine, reflecting current mutable state."""
        cfg = self._base
        return {
            'num_iterations': self.num_simulations,
            'c_puct_init': cfg.c_puct_init,
            'c_puct_base': cfg.c_puct_base,
            'c_puct_factor': cfg.c_puct_factor,
            'fpu_reduction_root': cfg.fpu_reduction_root,
            'fpu_reduction': cfg.fpu_reduction,
            'dirichlet_alpha': cfg.dirichlet_alpha,
            'dirichlet_epsilon': cfg.dirichlet_epsilon,
            'add_noise': True,
            'policy_softmax_temp': cfg.policy_softmax_temperature,
            'batch_size': cfg.batch_size,
            'smart_pruning': cfg.smart_pruning,
            'smart_pruning_factor': cfg.smart_pruning_factor,
            'two_fold_draw': cfg.two_fold_draw,
            'shaped_dirichlet': cfg.shaped_dirichlet,
            'uncertainty_weight': cfg.uncertainty_weight,
            'variance_scaling': cfg.variance_scaling,
            'contempt': cfg.contempt,
            'sibling_blending': getattr(cfg, 'sibling_blending', True),
            'nn_cache_size': cfg.nn_cache_size,
            'max_collapse_visits': getattr(cfg, 'max_collapse_visits', 8),
            'fpu_absolute_root': getattr(cfg, 'fpu_absolute_root', False),
            'fpu_absolute_root_value': getattr(cfg, 'fpu_absolute_root_value', 1.0),
            'mlh_weight': getattr(cfg, 'mlh_weight', 0.0),
            'mlh_cap': getattr(cfg, 'mlh_cap', 10.0),
            'mlh_q_threshold': getattr(cfg, 'mlh_q_threshold', 0.6),
        }


class CppMCTS:
    """Wrapper around C++ chess_mcts.SearchEngine matching Python MCTS API."""

    def __init__(self, model_path: str, mcts_config: MCTSConfig, device: str):
        self.config = _CppMCTSConfig(mcts_config)
        self.nn_cache = None  # Managed internally by C++ engine
        self._model_path = model_path
        self._device = device
        self._engine = chess_mcts.SearchEngine(
            model_path, device, self.config._to_dict()
        )

    def search(self, board: chess.Board) -> SearchResult:
        # Only push config if sims changed (avoid unnecessary Python→C++ call)
        if not hasattr(self, '_last_sims') or self._last_sims != self.config.num_simulations:
            self._engine.set_config({'num_iterations': self.config.num_simulations})
            self._last_sims = self.config.num_simulations

        # Pass current FEN directly — no need to replay entire game history.
        # For repetition detection, pass last 8 positions as history.
        fen = board.fen()
        history = []
        if len(board.move_stack) > 0:
            # Build short history for repetition detection (last 8 moves)
            temp = board.copy()
            recent_fens = []
            for _ in range(min(8, len(temp.move_stack))):
                temp.pop()
                recent_fens.append(temp.fen())
            # Start from oldest, replay forward
            if recent_fens:
                oldest_fen = recent_fens[-1]
                # Replay moves from oldest to current
                temp2 = chess.Board(oldest_fen)
                moves_to_replay = list(board.move_stack)[-len(recent_fens):]
                fen = oldest_fen
                history = [m.uci() for m in moves_to_replay]

        cpp_result = self._engine.search(fen, history)

        return SearchResult(
            best_move=chess.Move.from_uci(cpp_result.best_move) if cpp_result.best_move else None,
            visit_counts={
                chess.Move.from_uci(m): v
                for m, v in cpp_result.visit_counts.items()
            },
            root_value=cpp_result.root_value,
            policy_target=np.array(cpp_result.policy_target, dtype=np.float32),
            raw_policy=np.array(cpp_result.raw_policy, dtype=np.float32),
            raw_value=cpp_result.raw_value,
        )


@dataclass
class SelfPlayConfig:
    temperature_moves: int = 30
    max_moves: int = 512
    resign_threshold: float = -0.95
    consecutive_resign: int = 5
    q_ratio: float = 0.0  # Blend ratio: 0 = pure game result, 1 = pure search Q
    playout_cap_randomization: bool = True   # Alternate full/quick search
    playout_cap_fraction: float = 0.25       # Fraction of moves that get full search
    playout_cap_quick_sims: int = 100        # Simulations for quick search moves
    kld_adaptive: bool = True               # Adapt visit count based on KL divergence
    kld_min_sims: int = 100                 # Minimum simulations (used when KLD is low)
    kld_max_sims: int = 800                 # Maximum simulations (used when KLD is high)
    kld_threshold: float = 0.5              # KLD above this gets max sims
    syzygy_path: str | None = None          # Path to Syzygy tablebase files (None = disabled)
    random_opening_fraction: float = 0.05   # Fraction of games with random openings
    random_opening_moves: int = 8           # Max random moves for opening randomization
    opening_book_path: str | None = None    # File of starting FENs, one per line (None = disabled)
    opening_book_fraction: float = 0.5      # Fraction of games seeded from book FENs


@dataclass
class AdaptiveConfig:
    """Auto-tune sims/max_moves/games per generation for faster early training."""
    enabled: bool = True
    early_until: int = 5       # Generations 1..early_until use early settings
    mid_until: int = 15        # Generations (early_until+1)..mid_until interpolate
    early_sims: int = 100
    mid_sims: int = 200
    full_sims: int = 400
    early_max_moves: int = 150
    mid_max_moves: int = 300
    full_max_moves: int = 512
    early_games: int = 200
    mid_games: int = 100
    full_games: int = 50


def get_gen_settings(gen: int, adaptive: AdaptiveConfig) -> tuple[int, int, int]:
    """Returns (simulations, max_moves, games_per_gen) for this generation."""
    if not adaptive.enabled:
        return adaptive.full_sims, adaptive.full_max_moves, adaptive.full_games
    if gen <= adaptive.early_until:
        return adaptive.early_sims, adaptive.early_max_moves, adaptive.early_games
    elif gen <= adaptive.mid_until:
        t = (gen - adaptive.early_until) / (adaptive.mid_until - adaptive.early_until)
        sims = int(adaptive.early_sims + t * (adaptive.mid_sims - adaptive.early_sims))
        moves = int(adaptive.early_max_moves + t * (adaptive.mid_max_moves - adaptive.early_max_moves))
        games = int(adaptive.early_games + t * (adaptive.mid_games - adaptive.early_games))
        return sims, moves, games
    else:
        return adaptive.full_sims, adaptive.full_max_moves, adaptive.full_games


@dataclass
class GameRecord:
    planes: list[np.ndarray] = field(default_factory=list)
    policies: list[np.ndarray] = field(default_factory=list)
    values: list[np.ndarray] = field(default_factory=list)
    moves_left: list[float] = field(default_factory=list)
    surprise: list[float] = field(default_factory=list)
    use_policy: list[bool] = field(default_factory=list)  # True = full search, False = quick search
    result: str = '*'
    num_moves: int = 0
    moves_uci: list[str] = field(default_factory=list)


_opening_book_cache: dict[str, list[str]] = {}


def _load_opening_book(path: str | None) -> list[str]:
    """Load opening FENs from file. Cached by path. Returns [] if path is None/missing."""
    if not path:
        return []
    if path in _opening_book_cache:
        return _opening_book_cache[path]
    if not os.path.exists(path):
        print(f"  Warning: opening book not found at {path}, skipping")
        _opening_book_cache[path] = []
        return []
    with open(path) as f:
        fens = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    print(f"  Loaded {len(fens)} opening positions from {path}")
    _opening_book_cache[path] = fens
    return fens


def _sample_opening_fen(config: SelfPlayConfig) -> str | None:
    """Pick a random book FEN, or None if book not configured / not triggered."""
    if not config.opening_book_path or config.opening_book_fraction <= 0:
        return None
    if random.random() >= config.opening_book_fraction:
        return None
    fens = _load_opening_book(config.opening_book_path)
    if not fens:
        return None
    return random.choice(fens)


def _try_load_syzygy(path: str | None):
    """Try to load Syzygy tablebases. Returns tablebase object or None."""
    if path is None:
        return None
    try:
        import chess.syzygy
        tb = chess.syzygy.open_tablebase(path)
        return tb
    except Exception:
        return None


def rescore_with_tablebases(
    positions: list[tuple],
    board_history: list[chess.Board],
    tablebase,
) -> list[tuple]:
    """Rescore endgame positions using Syzygy tablebases.

    For positions with ≤5 pieces, replace the game-result WDL with the
    tablebase-correct result. Rescores backward from the first TB hit.
    """
    if tablebase is None:
        return positions

    # Find the earliest position where we have a TB result
    tb_wdl = [None] * len(positions)
    for i, board in enumerate(board_history):
        if chess.popcount(board.occupied) <= 5:
            try:
                wdl_val = tablebase.probe_wdl(board)
                # wdl_val: 2=win, 1=cursed win, 0=draw, -1=blessed loss, -2=loss
                # Convert to WDL from side-to-move perspective
                if wdl_val >= 1:
                    tb_wdl[i] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                elif wdl_val <= -1:
                    tb_wdl[i] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                else:
                    tb_wdl[i] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            except Exception:
                pass

    # Propagate TB results backward: once we hit a TB position, all prior
    # positions in the game get that result (adjusted for perspective)
    last_tb_wdl = None
    last_tb_stm = None
    for i in range(len(positions) - 1, -1, -1):
        if tb_wdl[i] is not None:
            last_tb_wdl = tb_wdl[i]
            last_tb_stm = board_history[i].turn
        if last_tb_wdl is not None:
            planes, policy_target, side_to_move, root_q, surprise, is_full = positions[i]
            # Adjust perspective if needed
            if side_to_move == last_tb_stm:
                corrected = last_tb_wdl.copy()
            else:
                corrected = last_tb_wdl[[2, 1, 0]].copy()  # flip win/loss
            positions[i] = (planes, policy_target, side_to_move, root_q, surprise, is_full, corrected)
        else:
            planes, policy_target, side_to_move, root_q, surprise, is_full = positions[i]
            positions[i] = (planes, policy_target, side_to_move, root_q, surprise, is_full, None)

    return positions


def play_game(mcts, config: SelfPlayConfig) -> GameRecord:
    """Play a single self-play game.

    Args:
        mcts: MCTS engine (Python MCTS or CppMCTS wrapper).
        config: Self-play configuration.
    """
    if mcts.nn_cache is not None:
        mcts.nn_cache.clear()

    # Opening book seeding: start from a random book position if configured.
    # Falls back to standard opening randomization when no book FEN is sampled.
    book_fen = _sample_opening_fen(config)
    if book_fen is not None:
        try:
            board = chess.Board(book_fen)
        except ValueError:
            board = chess.Board()
            book_fen = None
    else:
        board = chess.Board()
    record = GameRecord()

    # Opening randomization: play random moves to diversify openings (only if not book-seeded)
    if book_fen is None and config.random_opening_fraction > 0 and \
            random.random() < config.random_opening_fraction:
        n_random = random.randint(2, config.random_opening_moves)
        for _ in range(n_random):
            legal = list(board.legal_moves)
            if not legal or board.is_game_over():
                break
            board.push(random.choice(legal))
            record.moves_uci.append(board.peek().uci())

    positions = []  # (planes, policy_target, side_to_move, root_q, surprise, is_full)
    board_history = []  # board snapshots for tablebase rescoring
    resign_count = 0
    full_sims = mcts.config.num_simulations
    prev_kld = 0.0  # KL divergence from previous move (for adaptive visits)

    for move_num in range(config.max_moves):
        if board.is_game_over(claim_draw=True):
            break

        # Set temperature based on move number
        if move_num < config.temperature_moves:
            mcts.config.temperature = 1.0
        else:
            mcts.config.temperature = 0.01

        # Playout cap randomization: randomly choose full or quick search
        is_full_search = True
        if config.playout_cap_randomization:
            is_full_search = random.random() < config.playout_cap_fraction

        if is_full_search and config.kld_adaptive:
            # KLD-adaptive: scale sims based on previous move's KL divergence
            t = min(prev_kld / config.kld_threshold, 1.0)
            adaptive_sims = int(config.kld_min_sims + t * (config.kld_max_sims - config.kld_min_sims))
            mcts.config.num_simulations = adaptive_sims
        elif not is_full_search:
            mcts.config.num_simulations = config.playout_cap_quick_sims
        else:
            mcts.config.num_simulations = full_sims

        result = mcts.search(board)
        if result.best_move is None:
            break

        # Compute diff-focus surprise: how much MCTS changed the raw NN evaluation
        surprise = 0.0
        if result.raw_policy is not None:
            # Policy surprise: sum of |search_policy - raw_policy| over legal moves
            policy_diff = np.abs(result.policy_target - result.raw_policy)
            surprise += float(policy_diff.sum())
            # Value surprise: |root_value - raw_value|
            surprise += abs(result.root_value - result.raw_value)
            # KL divergence for adaptive visit count (next move)
            # KLD = sum(p * log(p / q)) where p = search policy, q = raw policy
            p = result.policy_target
            q = result.raw_policy
            mask = p > 1e-8
            if mask.any():
                prev_kld = float((p[mask] * np.log(p[mask] / (q[mask] + 1e-8))).sum())

        planes = encode_board(board)
        positions.append((planes, result.policy_target, board.turn, result.root_value, surprise, is_full_search))
        board_history.append(board.copy())

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

    # Restore full simulation count
    mcts.config.num_simulations = full_sims

    _finalize_record(record, board, positions, board_history, config)
    return record


def _finalize_record(
    record: GameRecord,
    board: chess.Board,
    positions: list,
    board_history: list,
    config: SelfPlayConfig,
) -> None:
    """Determine game result, rescore with tablebases, label positions with WDL."""
    # Determine game result if not already set (e.g. by resign logic)
    if record.result == '*':
        outcome = board.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            record.result = '1/2-1/2'
        elif outcome.winner == chess.WHITE:
            record.result = '1-0'
        else:
            record.result = '0-1'

    tablebase = _try_load_syzygy(config.syzygy_path)
    if tablebase is not None:
        positions = rescore_with_tablebases(positions, board_history, tablebase)
        tablebase.close()

    total_moves = len(positions)
    q_ratio = config.q_ratio

    for i, pos_data in enumerate(positions):
        if len(pos_data) == 7:
            planes, policy_target, side_to_move, root_q, pos_surprise, is_full, tb_wdl = pos_data
        else:
            planes, policy_target, side_to_move, root_q, pos_surprise, is_full = pos_data
            tb_wdl = None

        if tb_wdl is not None:
            z_wdl = tb_wdl
        elif record.result == '1/2-1/2':
            z_wdl = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        elif record.result == '1-0':
            if side_to_move == chess.WHITE:
                z_wdl = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            else:
                z_wdl = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:  # 0-1
            if side_to_move == chess.BLACK:
                z_wdl = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            else:
                z_wdl = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        if q_ratio > 0:
            q_win = max(0.0, root_q)
            q_loss = max(0.0, -root_q)
            q_draw = 1.0 - q_win - q_loss
            q_wdl = np.array([q_win, q_draw, q_loss], dtype=np.float32)
            wdl = (1.0 - q_ratio) * z_wdl + q_ratio * q_wdl
        else:
            wdl = z_wdl

        record.planes.append(planes)
        record.policies.append(policy_target)
        record.values.append(wdl)
        record.surprise.append(pos_surprise)
        record.moves_left.append(float(total_moves - i))
        record.use_policy.append(is_full)

    record.num_moves = total_moves


def play_games_batched(
    model_path: str,
    num_games: int,
    mcts_config: MCTSConfig,
    selfplay_config: SelfPlayConfig,
    device: str,
    parallel_games: int,
    use_fp16: bool = False,
    on_game_done=None,
    use_trt: bool = False,
    trt_engine_path: str = "",
) -> list[GameRecord]:
    """Play num_games concurrently using chess_mcts.GameManager.

    GameManager runs N games' MCTS in lockstep, batching NN evaluations across
    trees for a single shared GPU forward pass per step. Dynamic backfill keeps
    `parallel_games` active at all times until num_games is reached.

    Per-game playout-cap randomization and KLD-adaptive visits are NOT applied
    here — all moves use full sims (trade-off for cross-game batching).
    """
    cfg_proxy = _CppMCTSConfig(mcts_config)
    cfg_dict = cfg_proxy._to_dict()
    if use_trt:
        try:
            manager = chess_mcts.GameManager(
                model_path, device, cfg_dict, use_fp16, True, trt_engine_path,
            )
        except TypeError as e:
            raise RuntimeError(
                "chess_mcts module lacks use_trt support — rebuild with -DENABLE_TENSORRT=ON"
            ) from e
    else:
        try:
            manager = chess_mcts.GameManager(model_path, device, cfg_dict, use_fp16)
        except TypeError:
            # Older build without use_fp16 kwarg
            manager = chess_mcts.GameManager(model_path, device, cfg_dict)

    num_sims = mcts_config.num_simulations
    parallel_games = max(1, min(parallel_games, num_games))

    slot_board: list[chess.Board | None] = [None] * parallel_games
    slot_record: list[GameRecord | None] = [None] * parallel_games
    slot_positions: list[list] = [[] for _ in range(parallel_games)]
    slot_board_history: list[list] = [[] for _ in range(parallel_games)]
    slot_move_num = [0] * parallel_games
    slot_resign_count = [0] * parallel_games
    slot_start_time = [0.0] * parallel_games

    next_game_id = [0]
    completed: list[GameRecord] = []

    def start_slot(slot: int) -> bool:
        if next_game_id[0] >= num_games:
            slot_board[slot] = None
            return False
        next_game_id[0] += 1

        # Opening book seeding (falls back to standard board on invalid FEN)
        book_fen = _sample_opening_fen(selfplay_config)
        if book_fen is not None:
            try:
                board = chess.Board(book_fen)
            except ValueError:
                board = chess.Board()
                book_fen = None
        else:
            board = chess.Board()
        record = GameRecord()
        # Opening randomization (skip if book-seeded)
        if book_fen is None and selfplay_config.random_opening_fraction > 0 and \
                random.random() < selfplay_config.random_opening_fraction:
            n_random = random.randint(2, selfplay_config.random_opening_moves)
            for _ in range(n_random):
                legal = list(board.legal_moves)
                if not legal or board.is_game_over():
                    break
                mv = random.choice(legal)
                board.push(mv)
                record.moves_uci.append(mv.uci())
        slot_board[slot] = board
        slot_record[slot] = record
        slot_positions[slot] = []
        slot_board_history[slot] = []
        slot_move_num[slot] = 0
        slot_resign_count[slot] = 0
        slot_start_time[slot] = time.time()
        return True

    def finish_slot(slot: int) -> None:
        board = slot_board[slot]
        record = slot_record[slot]
        _finalize_record(record, board, slot_positions[slot],
                         slot_board_history[slot], selfplay_config)
        duration = time.time() - slot_start_time[slot]
        completed.append(record)
        if on_game_done is not None:
            on_game_done(record, duration, len(completed))

    for s in range(parallel_games):
        start_slot(s)

    while True:
        active = [s for s in range(parallel_games) if slot_board[s] is not None]
        if not active:
            break

        # Handle any games that ended before we search this round
        to_search: list[int] = []
        for s in active:
            board = slot_board[s]
            if slot_move_num[s] >= selfplay_config.max_moves or \
                    board.is_game_over(claim_draw=True):
                finish_slot(s)
                if not start_slot(s):
                    slot_board[s] = None
            else:
                to_search.append(s)
        if not to_search:
            continue

        # Build per-slot FEN + short history (last 8 moves) for repetition detection
        fens: list[str] = []
        histories: list[list[str]] = []
        for s in to_search:
            board = slot_board[s]
            if len(board.move_stack) > 0:
                temp = board.copy()
                n_hist = min(8, len(temp.move_stack))
                popped = [temp.pop().uci() for _ in range(n_hist)]
                fens.append(temp.fen())
                histories.append(list(reversed(popped)))
            else:
                fens.append(board.fen())
                histories.append([])

        manager.init_games_from_fen(fens, histories, num_sims)
        while not manager.all_complete():
            manager.step()

        for j, s in enumerate(to_search):
            board = slot_board[s]
            record = slot_record[s]
            result = manager.get_result(j)
            vc = result.visit_counts

            if not vc:
                # No legal moves — declare draw and finish
                record.result = '1/2-1/2'
                finish_slot(s)
                if not start_slot(s):
                    slot_board[s] = None
                continue

            # Temperature-based move selection over visit counts
            temperature = 1.0 if slot_move_num[s] < selfplay_config.temperature_moves else 0.01
            moves_list = list(vc.keys())
            visits = np.array([vc[m] for m in moves_list], dtype=np.float64)
            if temperature <= 0.01 or visits.sum() == 0:
                best_idx = int(np.argmax(visits))
            else:
                adj = visits ** (1.0 / temperature)
                probs = adj / adj.sum()
                best_idx = int(np.random.choice(len(moves_list), p=probs))
            move_uci = moves_list[best_idx]
            best_move = chess.Move.from_uci(move_uci)

            policy_target = np.asarray(result.policy_target, dtype=np.float32)
            raw_policy = np.asarray(result.raw_policy, dtype=np.float32)
            surprise = 0.0
            if raw_policy.size > 0 and raw_policy.shape == policy_target.shape:
                surprise += float(np.abs(policy_target - raw_policy).sum())
                surprise += abs(result.root_value - result.raw_value)

            planes = encode_board(board)
            slot_positions[s].append(
                (planes, policy_target, board.turn, result.root_value, surprise, True)
            )
            slot_board_history[s].append(board.copy())

            if result.root_value < selfplay_config.resign_threshold:
                slot_resign_count[s] += 1
                if slot_resign_count[s] >= selfplay_config.consecutive_resign:
                    record.result = '0-1' if board.turn == chess.WHITE else '1-0'
                    finish_slot(s)
                    if not start_slot(s):
                        slot_board[s] = None
                    continue
            else:
                slot_resign_count[s] = 0

            record.moves_uci.append(move_uci)
            board.push(best_move)
            slot_move_num[s] += 1

    return completed


def generate_games(
    model: ChessNetwork,
    num_games: int,
    output_path: str,
    mcts_config: MCTSConfig = MCTSConfig(),
    selfplay_config: SelfPlayConfig = SelfPlayConfig(),
    device: str = 'cpu',
    metrics_logger=None,
    model_path: str | None = None,
    parallel_games: int = 16,
    use_trt: bool = False,
    trt_engine_path: str = "",
) -> int:
    """Generate self-play games and save as .npz file.

    If C++ MCTS bindings are available and model_path is provided, uses the
    faster C++ GameManager with cross-game NN batching. Otherwise falls back to
    sequential Python MCTS.
    """
    use_cpp = HAS_CPP_MCTS and model_path is not None

    if use_cpp:
        # Use larger batch size for C++ MCTS to reduce GPU call overhead.
        # Preserve all other mcts_config fields (FPU, MLH, contempt, etc.) — earlier
        # code built a fresh MCTSConfig that silently dropped them.
        from dataclasses import replace
        cpp_config = replace(mcts_config, batch_size=max(mcts_config.batch_size, 128))
        print(f"  Using C++ GameManager ({device}, parallel_games={parallel_games}, "
              f"batch_size={cpp_config.batch_size})")
    else:
        if not HAS_CPP_MCTS:
            print("  Warning: C++ MCTS not available, using slower Python MCTS. "
                  "Build with -DBUILD_PYTHON=ON -DENABLE_NEURAL=ON for 50-100x speedup.")
        model = model.to(device)
        model.eval()

    all_planes = []
    all_policies = []
    all_values = []
    all_moves_left = []
    all_surprise = []
    all_use_policy = []

    start = time.time()

    if use_cpp:
        def _on_done(record, duration, completed_count):
            if metrics_logger is not None:
                from training.metrics import GameMetrics
                metrics_logger.record_game(GameMetrics(
                    game_num=completed_count,
                    num_moves=record.num_moves,
                    result=record.result,
                    duration_s=duration,
                    moves_uci=record.moves_uci,
                ))
            all_planes.extend(record.planes)
            all_policies.extend(record.policies)
            all_values.extend(record.values)
            all_moves_left.extend(record.moves_left)
            all_surprise.extend(record.surprise)
            all_use_policy.extend(record.use_policy)
            print(
                f"  Game {completed_count}/{num_games}: "
                f"{record.num_moves} moves, {record.result}, "
                f"{duration:.1f}s ({len(all_planes)} positions total)"
            )

        play_games_batched(
            model_path=model_path,
            num_games=num_games,
            mcts_config=cpp_config,
            selfplay_config=selfplay_config,
            device=device,
            parallel_games=parallel_games,
            use_fp16=(device == 'cuda'),
            on_game_done=_on_done,
            use_trt=use_trt,
            trt_engine_path=trt_engine_path,
        )
    else:
        mcts = MCTS(model, mcts_config, device=device)
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
            all_moves_left.extend(record.moves_left)
            all_surprise.extend(record.surprise)
            all_use_policy.extend(record.use_policy)

            total_positions = len(all_planes)
            print(
                f"  Game {game_num + 1}/{num_games}: "
                f"{record.num_moves} moves, {record.result}, "
                f"{game_time:.1f}s ({total_positions} positions total)"
            )

    elapsed = time.time() - start
    total_positions = len(all_planes)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    # Pack dense planes → 104 uint64 bitboards + 4 scalar metadata arrays.
    # Disk footprint is ~3–4× smaller (binary planes only, no scalar broadcast),
    # and np.savez_compressed adds gzip on top. The 8 scalar feature planes
    # (color/fullmove/castling/rule50/bias) are reconstructible from metadata
    # at load time, so we drop them from disk.
    from training.dataset import _pack_dense_planes, _extract_metadata_from_dense
    planes_arr = np.asarray(all_planes, dtype=np.float32)
    bitboards = _pack_dense_planes(planes_arr)
    meta = _extract_metadata_from_dense(planes_arr)
    # Release the dense array ASAP to keep peak RAM flat during save.
    del planes_arr

    np.savez_compressed(
        output_path,
        format_version=np.uint8(2),
        bitboards=bitboards,
        stm=meta['stm'],
        castling=meta['castling'],
        rule50=meta['rule50'],
        fullmove=meta['fullmove'],
        policies=np.array(all_policies, dtype=np.float32),
        values=np.array(all_values, dtype=np.float32),
        moves_left=np.array(all_moves_left, dtype=np.float32),
        surprise=np.array(all_surprise, dtype=np.float32),
        use_policy=np.array(all_use_policy, dtype=np.bool_),
    )

    print(f"Generated {num_games} games, {total_positions} positions in {elapsed:.1f}s")
    print(f"Saved to {output_path}")
    return total_positions


def resolve_tier(
    gen: int,
    schedule: list[tuple[int, int, int]] | None,
    default_blocks: int,
    default_filters: int,
) -> tuple[int, int]:
    """Return (blocks, filters) active for `gen` given a tier schedule.

    Schedule entries are (gen_start, blocks, filters) — last tier with
    gen_start <= gen wins. If schedule is falsy, returns the defaults.
    """
    if not schedule:
        return default_blocks, default_filters
    active = [t for t in schedule if t[0] <= gen]
    if not active:
        return default_blocks, default_filters
    _, blocks, filters = active[-1]
    return blocks, filters


def load_checkpoint_with_config(
    resume_from: str,
    default_blocks: int,
    default_filters: int,
    device: str,
) -> tuple[ChessNetwork, NetworkConfig, int]:
    """Load a checkpoint, honoring its saved NetworkConfig when present.

    Returns (model loaded with saved weights, its config, next_gen).
    When the checkpoint carries no 'config' key, falls back to the defaults
    supplied by the caller.
    """
    checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
    saved_cfg = checkpoint.get('config')
    if saved_cfg is not None:
        config = saved_cfg
    else:
        config = NetworkConfig(num_blocks=default_blocks, num_filters=default_filters)
    model = ChessNetwork(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    start_gen = checkpoint['generation'] + 1
    return model, config, start_gen


def _trt_available() -> bool:
    """Return True if both the tensorrt Python package and TRT bindings in
    chess_mcts are usable. Cheap probe — only imports tensorrt.
    """
    try:
        # build_trt_engine handles $TENSORRT_PATH DLL discovery on Windows
        from training import build_trt_engine  # noqa: F401
        import tensorrt  # noqa: F401
        return True
    except Exception:
        return False


def _build_trt_engine_for_self_play(model, cpp_model_path: str) -> str:
    """Export ONNX from `model` and build a TensorRT engine alongside cpp_model_path.

    Returns the .trt path. Uses env TENSORRT_PATH for DLL discovery (handled in
    training.build_trt_engine).
    """
    from training.export import export_onnx
    from training.build_trt_engine import build_engine

    onnx_path = cpp_model_path.replace('.pt', '.onnx')
    trt_path = cpp_model_path.replace('.pt', '.trt')
    export_onnx(model, onnx_path)
    build_engine(onnx_path, trt_path)
    return trt_path


def training_loop(
    generations: int = 10,
    games_per_gen: int = 400,
    train_epochs: int = 5,
    batch_size: int = 2048,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    num_simulations: int = 400,
    blocks: int = 10,
    filters: int = 128,
    window_size: int = 5,
    output_dir: str = 'models/current_run',
    device: str = 'auto',
    max_moves: int = 512,
    resign_threshold: float = -0.95,
    use_swa: bool = True,
    syzygy_path: str | None = None,
    opening_book_path: str | None = None,
    opening_book_fraction: float = 0.5,
    adaptive: AdaptiveConfig | None = None,
    resume_from: str | None = None,
    parallel_games: int = 128,
    network_schedule: list[tuple[int, int, int]] | None = None,
    use_trt: bool = True,
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
    # TRT fallback: if the caller opted into use_trt but the environment can't
    # provide it (no tensorrt wheel, TENSORRT_PATH unset, or chess_mcts built
    # without HAS_TENSORRT), warn once and continue with LibTorch FP16.
    if use_trt and not _trt_available():
        print("  [warn] use_trt=True but tensorrt not importable — falling back to LibTorch FP16. "
              "Install the tensorrt wheel from your TensorRT SDK and set TENSORRT_PATH to enable.")
        use_trt = False

    from torch.utils.data import DataLoader, WeightedRandomSampler
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

    # Resume path takes priority: saved architecture wins over caller args.
    start_gen = 1
    if resume_from:
        print(f"Resuming from checkpoint: {resume_from}")
        model, config, start_gen = load_checkpoint_with_config(
            resume_from,
            default_blocks=blocks,
            default_filters=filters,
            device=device,
        )
        blocks = config.num_blocks
        filters = config.num_filters
        optimizer = create_optimizer(model, lr=lr, weight_decay=weight_decay)
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except (ValueError, KeyError) as e:
            print(f"  Optimizer state could not be restored ({e}); continuing with fresh optimizer.")
        print(f"  Loaded generation {start_gen - 1} ({blocks}b{filters}f), resuming from gen {start_gen}")
    else:
        # Pick initial tier if schedule is set.
        blocks, filters = resolve_tier(1, network_schedule, blocks, filters)
        config = NetworkConfig(num_blocks=blocks, num_filters=filters)
        model = ChessNetwork(config).to(device)
        optimizer = create_optimizer(model, lr=lr, weight_decay=weight_decay)

    # Stochastic Weight Averaging: use averaged model for self-play (smoother policy)
    swa_model = None
    if use_swa:
        from torch.optim.swa_utils import AveragedModel
        swa_model = AveragedModel(model)

    # Enable Lc0-style search features for the training loop:
    # - Absolute root FPU: unvisited root children treated as value=1.0 (pessimistic),
    #   encouraging broader early root exploration.
    # - MLH PUCT bonus: in winning/losing positions, prefer children that shorten/lengthen
    #   the game. Disabled until MLH outputs diverge (which requires training signal).
    mcts_config = MCTSConfig(
        num_simulations=num_simulations,
        fpu_absolute_root=True,
        fpu_absolute_root_value=1.0,
        mlh_weight=0.03,
        mlh_cap=8.0,
        mlh_q_threshold=0.6,
    )
    selfplay_config = SelfPlayConfig(
        max_moves=max_moves,
        resign_threshold=resign_threshold,
        syzygy_path=syzygy_path,
        opening_book_path=opening_book_path,
        opening_book_fraction=opening_book_fraction,
    )

    # Default adaptive config if not provided
    if adaptive is None:
        adaptive = AdaptiveConfig(enabled=False, full_sims=num_simulations,
                                   full_max_moves=max_moves, full_games=games_per_gen)

    end_gen = start_gen + generations - 1
    print(f"Starting training loop: generations {start_gen}-{end_gen}, {games_per_gen} games/gen")
    print(f"Device: {device}, Model: {blocks} blocks, {filters} filters"
          f"{', SWA enabled' if use_swa else ''}"
          f"{', adaptive' if adaptive.enabled else ''}"
          f"{f', tiered schedule={network_schedule}' if network_schedule else ''}")

    for gen in range(start_gen, end_gen + 1):
        gen_start = time.time()

        # Tier transition: rebuild model (and optimizer/SWA) if the schedule
        # says this generation belongs to a new tier.
        target_blocks, target_filters = resolve_tier(
            gen, network_schedule, blocks, filters,
        )
        if (target_blocks, target_filters) != (config.num_blocks, config.num_filters):
            print(f"  Scaling net: {config.num_blocks}b{config.num_filters}"
                  f" -> {target_blocks}b{target_filters}")
            config = NetworkConfig(num_blocks=target_blocks, num_filters=target_filters)
            model = ChessNetwork(config).to(device)
            optimizer = create_optimizer(model, lr=lr, weight_decay=weight_decay)
            if use_swa:
                from torch.optim.swa_utils import AveragedModel
                swa_model = AveragedModel(model)
            blocks, filters = target_blocks, target_filters

        # Adaptive settings: adjust sims/max_moves/games per generation
        gen_sims, gen_max_moves, gen_games = get_gen_settings(gen, adaptive)
        mcts_config.num_simulations = gen_sims
        selfplay_config.max_moves = gen_max_moves

        print(f"\n{'='*60}")
        print(f"Generation {gen}/{end_gen}"
              f" (sims={gen_sims}, max_moves={gen_max_moves}, games={gen_games},"
              f" net={blocks}b{filters}f)")
        print(f"{'='*60}")

        # 1. Generate self-play games (use SWA model if available)
        data_path = os.path.join(data_dir, f'gen_{gen:03d}.npz')
        play_model = swa_model if (swa_model is not None and gen > 1) else model
        play_model.eval()

        # Export TorchScript model for C++ MCTS (always export so C++ engine can use it)
        # Always export from the base model (SWA wrapper lacks .config)
        cpp_model_path = os.path.join(checkpoint_dir, 'selfplay_model.pt')
        trt_engine_path = ""
        if swa_model is not None and gen > 1:
            # Copy SWA params into base model temporarily for export
            swa_state = swa_model.module.state_dict()
            orig_state = model.state_dict()
            model.load_state_dict(swa_state)
            export_torchscript(model, cpp_model_path, device=device)
            if use_trt:
                trt_engine_path = _build_trt_engine_for_self_play(model, cpp_model_path)
            model.load_state_dict(orig_state)
        else:
            export_torchscript(model, cpp_model_path, device=device)
            if use_trt:
                trt_engine_path = _build_trt_engine_for_self_play(model, cpp_model_path)

        num_positions = generate_games(
            play_model, gen_games, data_path,
            mcts_config=mcts_config,
            selfplay_config=selfplay_config,
            device=device,
            metrics_logger=metrics_logger,
            model_path=cpp_model_path,
            parallel_games=parallel_games,
            use_trt=use_trt,
            trt_engine_path=trt_engine_path,
        )

        # 2. Load sliding window of recent generations
        window_start = max(1, gen - window_size + 1)
        npz_paths = [
            os.path.join(data_dir, f'gen_{g:03d}.npz')
            for g in range(window_start, gen + 1)
        ]
        dataset = ChessDataset(npz_paths, mirror=True)
        if dataset.surprise_weights is not None:
            # Diff-focus sampling: oversample positions where MCTS disagreed with raw NN
            weights = dataset.surprise_weights + 1e-6  # Avoid zero weights
            sampler = WeightedRandomSampler(weights, len(dataset), replacement=True)
            dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=True)
        else:
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        # 3. Train for train_epochs epochs (with mixed precision on CUDA)
        model.train()
        total_loss_sum = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        num_batches = 0

        use_amp = (device == 'cuda')
        scaler = torch.amp.GradScaler(enabled=use_amp)

        for epoch in range(train_epochs):
            for batch in dataloader:
                planes = batch[0].to(device)
                policy_target = batch[1].to(device)
                value_target = batch[2].to(device)
                mlh_target = None
                policy_mask = None
                bi = 3
                if bi < len(batch) and batch[bi].dtype in (torch.float32, torch.float64):
                    mlh_target = batch[bi].to(device)
                    bi += 1
                if bi < len(batch) and batch[bi].dtype == torch.bool:
                    policy_mask = batch[bi].to(device)

                optimizer.zero_grad()
                with torch.amp.autocast(device_type=device, enabled=use_amp):
                    policy_logits, value_logits, mlh_pred = model(planes)
                    total_loss, policy_loss, value_loss = compute_loss(
                        policy_logits, value_logits, policy_target, value_target,
                        mlh_pred, mlh_target, policy_mask,
                    )
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()

                total_loss_sum += total_loss.item()
                policy_loss_sum += policy_loss.item()
                value_loss_sum += value_loss.item()
                num_batches += 1

        # Update SWA model after training
        if swa_model is not None:
            swa_model.update_parameters(model)

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
            network={'blocks': config.num_blocks, 'filters': config.num_filters},
        )

    # Export final TorchScript model
    final_path = os.path.join(output_dir, 'model_final.pt')
    export_torchscript(model, final_path, device=device)
    print(f"\nTraining complete. Final model exported to {final_path}")


def main():
    parser = argparse.ArgumentParser(description='Self-play training for chess AI')
    subparsers = parser.add_subparsers(dest='command')

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
    loop_parser.add_argument('--games-per-gen', type=int, default=400)
    loop_parser.add_argument('--train-epochs', type=int, default=5)
    loop_parser.add_argument('--batch-size', type=int, default=2048)
    loop_parser.add_argument('--lr', type=float, default=1e-3)
    loop_parser.add_argument('--simulations', type=int, default=400)
    loop_parser.add_argument('--blocks', type=int, default=10)
    loop_parser.add_argument('--filters', type=int, default=128)
    loop_parser.add_argument('--window-size', type=int, default=5)
    loop_parser.add_argument('--output-dir', type=str, default='models/current_run')
    loop_parser.add_argument('--device', type=str, default='auto')
    loop_parser.add_argument('--max-moves', type=int, default=512)
    loop_parser.add_argument('--syzygy', type=str, default=None, help='Path to Syzygy tablebase files')
    loop_parser.add_argument('--opening-book', type=str, default=None, help='Path to opening book FEN file (one FEN per line)')
    loop_parser.add_argument('--opening-book-fraction', type=float, default=0.5, help='Fraction of games seeded from book')
    loop_parser.add_argument('--adaptive', action='store_true', default=False, help='Enable adaptive settings per generation')
    loop_parser.add_argument('--no-adaptive', dest='adaptive', action='store_false')
    loop_parser.add_argument('--early-sims', type=int, default=100, help='Simulations for early generations')
    loop_parser.add_argument('--early-max-moves', type=int, default=150, help='Max moves for early generations')
    loop_parser.add_argument('--early-games', type=int, default=200, help='Games per early generation')
    loop_parser.add_argument('--resume-from', type=str, default=None, help='Resume from a checkpoint file (e.g. selfplay_output/checkpoints/model_gen_5.pt)')
    loop_parser.add_argument('--parallel-games', type=int, default=128, help='Concurrent self-play games (C++ GameManager cross-game batching)')
    loop_parser.add_argument('--use-trt', dest='use_trt', action='store_true', default=True, help='Export ONNX + build TensorRT engine per generation and run self-play through the TRT backend (default: on)')
    loop_parser.add_argument('--no-use-trt', dest='use_trt', action='store_false', help='Disable TensorRT backend (use LibTorch only)')

    args = parser.parse_args()

    # Default to 'loop' if no subcommand given
    if args.command is None:
        args.command = 'loop'
        args = loop_parser.parse_args()
        args.command = 'loop'

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
        adaptive_config = None
        if getattr(args, 'adaptive', False):
            adaptive_config = AdaptiveConfig(
                enabled=True,
                early_sims=getattr(args, 'early_sims', 100),
                early_max_moves=getattr(args, 'early_max_moves', 150),
                early_games=getattr(args, 'early_games', 200),
                full_sims=args.simulations,
                full_max_moves=args.max_moves,
                full_games=args.games_per_gen,
            )
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
            syzygy_path=getattr(args, 'syzygy', None),
            opening_book_path=getattr(args, 'opening_book', None),
            opening_book_fraction=getattr(args, 'opening_book_fraction', 0.5),
            adaptive=adaptive_config,
            resume_from=getattr(args, 'resume_from', None),
            parallel_games=getattr(args, 'parallel_games', 128),
            use_trt=getattr(args, 'use_trt', True),
        )


if __name__ == '__main__':
    main()
