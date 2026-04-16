"""Self-play game generation and reinforcement learning loop.

Usage:
    python -m training.selfplay generate --games 50 --simulations 100
    python -m training.selfplay loop --generations 10 --games-per-gen 50
"""

import argparse
import os
import random
import time
from dataclasses import dataclass, field, asdict

import chess
import numpy as np
import torch

from training.config import NetworkConfig, SelfPlayConfig as _SelfPlayConfigBase
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
            'use_syzygy': True,
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


# SelfPlayConfig canonically lives in training.config as of the Stage 2
# Lc0-parity refactor. Re-exported here for backward compat with callers that
# still do `from training.selfplay import SelfPlayConfig`.
SelfPlayConfig = _SelfPlayConfigBase


@dataclass
class AdaptiveConfig:
    """Auto-tune sims/max_moves/games per generation for faster early training.

    Defaults are tuned for RTX 3080 (10GB) + 8-core/16-thread CPU + 32GB RAM
    with a pretrained network (~1600 Elo from human games). The pretrained
    network already plays reasonable chess, so early sims can be higher than
    train-from-scratch settings to avoid garbage data.
    """
    enabled: bool = True
    early_until: int = 5       # Generations 1..early_until use early settings
    mid_until: int = 15        # Generations (early_until+1)..mid_until interpolate
    early_sims: int = 400
    mid_sims: int = 400
    full_sims: int = 400
    early_max_moves: int = 300
    mid_max_moves: int = 400
    full_max_moves: int = 512
    early_games: int = 512
    mid_games: int = 512
    full_games: int = 512


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
    # v2 decomposed eval signals (Stage 3). One WDL triple per position, so the
    # trainer can blend game_result / best / played / raw_nn at load time
    # instead of being stuck with a baked-in q_ratio. Python fallback path
    # populates approximations via _q_to_wdl; C++ path carries real values
    # through _loop_record_to_legacy.
    best_eval: list[np.ndarray] = field(default_factory=list)     # (3,) per row
    played_eval: list[np.ndarray] = field(default_factory=list)   # (3,) per row
    raw_nn_eval: list[np.ndarray] = field(default_factory=list)   # (3,) per row
    adjudicated: list[bool] = field(default_factory=list)
    was_playthrough: list[bool] = field(default_factory=list)


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


def _choose_starting_fen(cfg, opening_book_fens, pool, rng):
    """Decide a new game's starting FEN + seed_source label.

    Priority: DiscardPool (with probability cfg.discarded_start_chance, filtered
    to ≥ cfg.discarded_min_pieces) → opening book (if provided) → startpos.
    Pops from the pool up to 3 times to skip under-pieced entries; if nothing
    qualifies, falls back without consuming further pool entries.

    Returns (fen, seed_source) where seed_source ∈ {"discard_pool",
    "opening_book", "standard"}.
    """
    if pool is not None and rng.random() < cfg.discarded_start_chance:
        for _ in range(3):
            fen = pool.pop()
            if fen is None:
                break
            piece_count = sum(1 for ch in fen.split()[0] if ch.isalpha())
            if piece_count >= cfg.discarded_min_pieces:
                return fen, "discard_pool"
    if opening_book_fens:
        return rng.choice(opening_book_fens), "opening_book"
    return chess.STARTING_FEN, "standard"


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
        # v2 decomposed signals (Python fallback): no real per-child/root WDL
        # available, so approximate best/played/raw_nn from the same root_q
        # used for q_wdl above. Trainer blend will converge on `game_result`
        # weighting for this path unless/until we wire real WDLs through.
        q_win = max(0.0, root_q)
        q_loss = max(0.0, -root_q)
        q_draw = 1.0 - q_win - q_loss
        approx_wdl = np.array([q_win, q_draw, q_loss], dtype=np.float32)
        record.best_eval.append(approx_wdl)
        record.played_eval.append(approx_wdl)
        record.raw_nn_eval.append(approx_wdl)
        record.adjudicated.append(False)
        record.was_playthrough.append(False)

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
    discard_pool=None,
    continuous_flow: bool = True,
) -> list[GameRecord]:
    """Play num_games concurrently using chess_mcts.GameManager + GameLoopManager.

    Stage 2 Lc0-parity refactor: Python (GameLoopManager) drives per-step
    orchestration — temperature, playout-cap randomization, KLD-adaptive
    target sims, ply-cap adjudication. C++ GameManager is a pure search
    engine exposing `step_stats` / `apply_move`.

    The on-disk format is preserved (Stage 3 will replace the schema). Some
    Stage-1-era features (resign, opening book, random-opening seeding,
    surprise weights) are temporarily simplified — they'll be restored in
    later stages:
      - Resign: dropped (Stage 4 will reintroduce with adjudication).
      - Opening book / random-opening: dropped for the C++ path (Stage 6).
      - Surprise: stamped as 0.0 (Stage 3 will recompute from soft-policy
        deltas once the v2 writer lands).
    """
    from training.selfplay_loop import GameLoopManager, GameRecord as LoopGameRecord

    cfg_proxy = _CppMCTSConfig(mcts_config)
    cfg_dict = cfg_proxy._to_dict()

    def _make_manager():
        if use_trt:
            try:
                return chess_mcts.GameManager(
                    model_path, device, cfg_dict, use_fp16, True, trt_engine_path,
                )
            except TypeError as e:
                raise RuntimeError(
                    "chess_mcts module lacks use_trt support — rebuild with -DENABLE_TENSORRT=ON"
                ) from e
        try:
            return chess_mcts.GameManager(model_path, device, cfg_dict, use_fp16)
        except TypeError:
            return chess_mcts.GameManager(model_path, device, cfg_dict)

    # Rebuild a fresh SelfPlayConfig view that wires Stage 2 fields from the
    # mcts_config / selfplay_config pair. We keep the legacy selfplay_config
    # object intact for _finalize_record (Syzygy, q_ratio, etc).
    full_sims = max(1, mcts_config.num_simulations)
    quick_sims = max(1, min(selfplay_config.playout_cap_quick_sims, full_sims))
    min_sims = max(1, min(selfplay_config.kld_min_sims, full_sims))
    # playout_cap_fraction was "fraction of FULL moves"; GameLoopManager uses
    # playout_cap_p = P(QUICK move). Convert accordingly, but only when the
    # legacy toggle is on.
    if selfplay_config.playout_cap_randomization:
        playout_cap_p = max(0.0, min(1.0, 1.0 - selfplay_config.playout_cap_fraction))
    else:
        playout_cap_p = 0.0
    # Plies per full move ≈ 2 * moves for temperature schedule (temperature_moves
    # is moves-based; opening_temp_plies is ply-based).
    opening_temp_plies = max(1, selfplay_config.temperature_moves * 2)
    kld_max = max(1, min(selfplay_config.kld_max_sims, full_sims))
    loop_cfg = _replace_selfplay_cfg(
        selfplay_config,
        num_games=parallel_games,
        full_sims=full_sims,
        quick_sims=quick_sims,
        min_sims=min_sims,
        playout_cap_p=playout_cap_p,
        opening_temp=0.6,
        opening_temp_plies=16,      # ~8 moves of exploration, then decay
        temp_floor=0.1,
        temp_decay_plies=10,        # smooth decay over 5 moves after opening
        use_kld_adaptive=selfplay_config.kld_adaptive,
        kld_threshold=selfplay_config.kld_threshold,
        kld_max_sims=kld_max,
        # max_ply converts from move-count to ply-count.
        max_ply=max(1, selfplay_config.max_moves * 2),
    )

    parallel_games = max(1, min(parallel_games, num_games))
    completed: list[GameRecord] = []

    # Load opening book FENs once for all games (shared by both flow modes).
    book_fens = _load_opening_book(selfplay_config.opening_book_path)

    if continuous_flow:
        # Continuous-flow mode: one manager, one pool, slots respawn as games
        # finish. Eliminates the batch-boundary tail-latency cost.
        from training.selfplay_loop import GamePoolManager
        manager = _make_manager()
        manager.init_games(parallel_games, full_sims)
        pool_cfg = _replace_selfplay_cfg(loop_cfg, num_games=parallel_games)
        pool = GamePoolManager(
            manager, pool_cfg, discard_pool=discard_pool, rng_seed=None,
            opening_book_fens=book_fens,
        )
        pool_start = time.time()

        def _on_pool_done(lrec, completion_idx):
            record = _loop_record_to_legacy(lrec, selfplay_config)
            # Per-game duration is not cleanly attributable under continuous
            # flow (slots are reused). Report cumulative elapsed / games-done
            # so the external on_game_done callback still gets a monotonic
            # signal for progress logging.
            duration = (time.time() - pool_start) / max(1, completion_idx)
            completed.append(record)
            if on_game_done is not None:
                on_game_done(record, duration, len(completed))

        pool.run_pool(num_games, on_game_done=_on_pool_done)
    else:
        # Legacy batch-boundary mode: kept so `--continuous-flow` can be
        # turned off for side-by-side comparison. Remove once flow mode has
        # been stable for several generations.
        remaining = num_games

        # We run in rounds of `parallel_games` games at a time. Each round builds
        # a fresh GameManager and runs the loop to completion.
        while remaining > 0:
            batch = min(parallel_games, remaining)
            manager = _make_manager()
            manager.init_games(batch, full_sims)

            # Rebind the loop cfg to match the actual game count in this batch.
            batch_cfg = _replace_selfplay_cfg(loop_cfg, num_games=batch)
            loop = GameLoopManager(manager, batch_cfg, discard_pool=discard_pool, rng_seed=None,
                                   opening_book_fens=book_fens)
            start_times = [time.time()] * batch
            loop_records: list[LoopGameRecord] = loop.run_until_all_complete()

            for j, lrec in enumerate(loop_records):
                record = _loop_record_to_legacy(lrec, selfplay_config)
                duration = time.time() - start_times[j]
                completed.append(record)
                if on_game_done is not None:
                    on_game_done(record, duration, len(completed))

            remaining -= batch

    return completed


def _replace_selfplay_cfg(cfg: SelfPlayConfig, **overrides) -> SelfPlayConfig:
    """Return a copy of cfg with the given Stage 2 / loop fields overridden."""
    from dataclasses import replace
    return replace(cfg, **overrides)


def _loop_record_to_legacy(
    lrec, selfplay_config: SelfPlayConfig
) -> GameRecord:
    """Project a Stage 2 selfplay_loop.GameRecord into the legacy
    training.selfplay.GameRecord format used by on-disk .npz writers.

    Preserves the existing .npz schema — Stage 3 will replace this with a v2
    writer that carries best_eval/played_eval/raw_nn_eval directly.
    """
    record = GameRecord()

    # Replay the game board-by-board so we can (a) encode planes for each
    # position, (b) build board_history for Syzygy rescoring, (c) resolve
    # final result based on board.outcome() + terminal_status, and (d) map
    # the per-legal-move visit distribution back to 1858-dim indices.
    positions: list[tuple] = []
    board_history: list = []
    # Parallel arrays for v2 decomposed eval signals (populated in lock-step
    # with `positions`). Zipped with _finalize_record's output after the fact.
    loop_best: list[np.ndarray] = []
    loop_played: list[np.ndarray] = []
    loop_raw_nn: list[np.ndarray] = []
    loop_adjudicated: list[bool] = []
    loop_was_playthrough: list[bool] = []

    for row in lrec.rows:
        try:
            board = chess.Board(row.fen)
        except ValueError:
            continue
        planes = encode_board(board)

        # Project visits (over row.legal_moves_uci) into 1858-dim policy space.
        policy_target = np.zeros(POLICY_SIZE, dtype=np.float32)
        if row.legal_moves_uci and row.visits_policy:
            for uci, prob in zip(row.legal_moves_uci, row.visits_policy):
                try:
                    mv = chess.Move.from_uci(uci)
                except ValueError:
                    continue
                idx = _uci_to_policy_index(mv, board.turn)
                if idx is not None and 0 <= idx < POLICY_SIZE:
                    policy_target[idx] += float(prob)
            s = float(policy_target.sum())
            if s > 0:
                policy_target /= s

        # Scalar root Q approximation: played_eval (w,d,l) → w - l. Preserved
        # for legacy q-blend downstream; v2 fields below carry the real WDLs.
        w, _d, l = row.played_eval
        root_q = float(w - l)

        # Surprise stamped 0.0 for Stage 2; Stage 3 will recompute from
        # soft-policy deltas.
        surprise = 0.0
        is_full = bool(row.is_full_search)

        positions.append((planes, policy_target, board.turn, root_q, surprise, is_full))
        board_history.append(board.copy())
        record.moves_uci.append(row.played_uci or "")
        loop_best.append(np.asarray(row.best_eval, dtype=np.float32))
        loop_played.append(np.asarray(row.played_eval, dtype=np.float32))
        loop_raw_nn.append(np.asarray(row.raw_nn_eval, dtype=np.float32))
        loop_adjudicated.append(bool(getattr(row, "adjudicated", False)))
        loop_was_playthrough.append(bool(row.was_playthrough))

    # Determine the final board state for result labeling. Replay moves
    # forward from startpos? The loop doesn't give us the actual moves
    # played, only FENs per position. For result determination we rely on
    # terminal_status from the loop's finalize step:
    if lrec.terminal_status == 1:
        # Side-to-move at terminal won; last recorded row's stm (if any)
        # determines color perspective. But when adjudicated as draw, that's
        # handled below. Map terminal_status=+1 to the color that moved last.
        # Without move info, we derive from the last position's board.turn
        # (which is the side to move at the terminal position).
        last_stm_white = (positions[-1][2] == chess.WHITE) if positions else True
        record.result = '1-0' if last_stm_white else '0-1'
    elif lrec.terminal_status == -1:
        last_stm_white = (positions[-1][2] == chess.WHITE) if positions else True
        record.result = '0-1' if last_stm_white else '1-0'
    else:
        record.result = '1/2-1/2'

    # Build a synthetic final board for _finalize_record's tablebase rescoring.
    # We use the last row's FEN (or startpos fallback).
    if lrec.rows:
        try:
            final_board = chess.Board(lrec.rows[-1].fen)
        except ValueError:
            final_board = chess.Board()
    else:
        final_board = chess.Board()

    _finalize_record(record, final_board, positions, board_history, selfplay_config)

    # Overwrite _finalize_record's q-approximations with the real per-row
    # decomposed eval signals the GameLoopManager already computed. Length
    # matches record.planes (both built in lock-step from lrec.rows).
    n = len(record.planes)
    if len(loop_best) == n:
        record.best_eval = loop_best
        record.played_eval = loop_played
        record.raw_nn_eval = loop_raw_nn
        record.adjudicated = loop_adjudicated
        record.was_playthrough = loop_was_playthrough
    return record


def _uci_to_policy_index(move: chess.Move, turn) -> int | None:
    """Map a chess.Move to the 1858-dim policy index from side-to-move POV.

    For black-to-move, the encoder flips the board via `square_mirror`, so
    we flip from/to squares before calling move_to_index.
    """
    from training.encoder import move_to_index
    fr = move.from_square
    to = move.to_square
    if turn == chess.BLACK:
        fr = chess.square_mirror(fr)
        to = chess.square_mirror(to)
    promo = None
    if move.promotion is not None and move.promotion != chess.QUEEN:
        promo_map = {chess.KNIGHT: 'n', chess.BISHOP: 'b', chess.ROOK: 'r'}
        promo = promo_map.get(move.promotion)
    return move_to_index(fr, to, promo)


def _extend_v2(
    record: "GameRecord",
    all_best_eval: list,
    all_played_eval: list,
    all_raw_nn_eval: list,
    all_adjudicated: list,
    all_was_playthrough: list,
) -> None:
    """Extend per-row v2 aggregates from a finalized GameRecord.

    Length-tolerant: if a fallback path populated fewer v2 rows than planes
    (older fake game managers in tests, etc.), pad with neutral drawn WDL so
    downstream shapes line up with all_planes.
    """
    n = len(record.planes)
    neutral = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    be = list(record.best_eval)
    pe = list(record.played_eval)
    rn = list(record.raw_nn_eval)
    adj = list(record.adjudicated)
    wp = list(record.was_playthrough)
    while len(be) < n: be.append(neutral)
    while len(pe) < n: pe.append(neutral)
    while len(rn) < n: rn.append(neutral)
    while len(adj) < n: adj.append(False)
    while len(wp) < n: wp.append(False)
    all_best_eval.extend(be[:n])
    all_played_eval.extend(pe[:n])
    all_raw_nn_eval.extend(rn[:n])
    all_adjudicated.extend(adj[:n])
    all_was_playthrough.extend(wp[:n])


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
    discard_pool=None,
    continuous_flow: bool = True,
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
        cpp_config = replace(mcts_config, batch_size=max(mcts_config.batch_size, 256))
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
    # v2 decomposed eval signals (Stage 3).
    all_best_eval = []
    all_played_eval = []
    all_raw_nn_eval = []
    all_adjudicated = []
    all_was_playthrough = []
    # Stage 7: per-game row counts so the calibrator can group playthrough
    # rows by game (aggregate shard has no native game boundary).
    all_game_lengths: list[int] = []

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
            _extend_v2(
                record, all_best_eval, all_played_eval, all_raw_nn_eval,
                all_adjudicated, all_was_playthrough,
            )
            all_game_lengths.append(len(record.planes))
            print(
                f"  Game {completed_count}/{num_games}: "
                f"{record.num_moves} plies, {record.result}, "
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
            discard_pool=discard_pool,
            continuous_flow=continuous_flow,
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
            _extend_v2(
                record, all_best_eval, all_played_eval, all_raw_nn_eval,
                all_adjudicated, all_was_playthrough,
            )
            all_game_lengths.append(len(record.planes))

            total_positions = len(all_planes)
            print(
                f"  Game {game_num + 1}/{num_games}: "
                f"{record.num_moves} plies, {record.result}, "
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

    def _stack_wdl(arrs, n):
        if not arrs:
            return np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float32), (n, 1))
        return np.stack([np.asarray(a, dtype=np.float32) for a in arrs])

    np.savez_compressed(
        output_path,
        format_version=np.uint8(2),
        schema_version=np.int32(2),
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
        # v2 decomposed eval signals — final_wdl lives in `values` already
        # (stamped by _finalize_record). best/played/raw_nn are the new
        # per-row arrays the trainer blends at load time.
        best_eval=_stack_wdl(all_best_eval, total_positions),
        played_eval=_stack_wdl(all_played_eval, total_positions),
        raw_nn_eval=_stack_wdl(all_raw_nn_eval, total_positions),
        adjudicated=np.asarray(all_adjudicated, dtype=np.bool_),
        was_playthrough=np.asarray(all_was_playthrough, dtype=np.bool_),
        game_lengths=np.asarray(all_game_lengths, dtype=np.int32),
    )

    print(f"Generated {num_games} games, {total_positions} positions in {elapsed:.1f}s",
          flush=True)
    print(f"Saved to {output_path}", flush=True)
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


def _build_trt_engine_for_self_play(
    model, cpp_model_path: str, max_batch: int = 512,
) -> str:
    """Export ONNX from `model` and build a fresh TensorRT engine.

    Follows Lc0's pattern: rebuild per weight-set, amortize via a persistent
    timing cache next to the engine file. Previously we refit in place; that
    path silently corrupted the engine because `torch.onnx.export` with
    `do_constant_folding=True` folds BN-fused conv weights into anonymous
    constants that are no longer refittable. See `build_trt_engine.py`
    module docstring.

    Returns the .trt path. Uses env TENSORRT_PATH for DLL discovery.
    """
    import gc as _gc
    import time as _time
    import torch as _torch
    from training.export import export_onnx
    from training.build_trt_engine import build_engine

    onnx_path = cpp_model_path.replace('.pt', '.onnx')
    trt_path = cpp_model_path.replace('.pt', '.trt')
    timing_cache_path = os.path.join(
        os.path.dirname(os.path.abspath(trt_path)) or '.', 'trt_timing.cache',
    )
    export_onnx(model, onnx_path)
    # TRT builder kernel autotune probes many candidate allocations; on a 10GB
    # GPU with training artifacts (optimizer states, SWA, cuDNN autotune cache)
    # still resident, those probes OOM. Free as much as we can first. Moving
    # the model to CPU during the build adds ~100MB of headroom; we restore
    # it below (caller still holds the same Module reference).
    orig_device = next(model.parameters()).device
    moved = False
    if _torch.cuda.is_available():
        if orig_device.type == 'cuda':
            model.to('cpu')
            moved = True
        _gc.collect()
        _torch.cuda.empty_cache()
    t0 = _time.time()
    # TRT max_batch must be >= the MCTS gather batch. opt_batch is set to half of
    # max so both small and large batches get decent kernels in one profile.
    opt_batch = max(128, max_batch // 2)
    # workspace_mb=1024 (down from 2048): on a 10GB GPU with training artifacts
    # (optimizer states, SWA copy, cuDNN autotune cache) resident at gen boundary,
    # a 2GB TRT workspace plus kernel-probe allocations OOMs. 1GB is ample for
    # a 20b×256f network and leaves headroom for the probing phase.
    build_engine(
        onnx_path, trt_path,
        opt_batch=opt_batch, max_batch=max_batch,
        workspace_mb=1024,
        timing_cache_path=timing_cache_path,
    )
    elapsed = _time.time() - t0
    print(f"  [trt] build took {elapsed:.1f}s")
    if moved:
        model.to(orig_device)
    return trt_path


def _build_window_dataloader(
    data_dir: str,
    window_start: int,
    window_end: int,
    batch_size: int,
    adjudicated_weight: float = 0.5,
):
    """Build a DataLoader over gen_{start..end}.npz with diff-focus sampling.

    Returns None if no usable npz files exist in the window (lets callers skip
    warm-up for early generations with no prior data).
    """
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from training.dataset import ChessDataset

    npz_paths = [
        os.path.join(data_dir, f'gen_{g:03d}.npz')
        for g in range(window_start, window_end + 1)
    ]
    npz_paths = [p for p in npz_paths if os.path.exists(p)]
    if not npz_paths:
        return None, None
    dataset = ChessDataset(npz_paths, mirror=True, adjudicated_weight=adjudicated_weight)
    # CPU-side bitboard unpack + plane assembly is the self-play loop's main
    # data-prep cost. With num_workers=0 the GPU sat idle between batches
    # (CPU at ~6% on a 16-thread box). 4 workers parallelize unpacking,
    # persistent_workers=True amortizes spawn cost across train_epochs,
    # pin_memory speeds the host→device copy.
    _common = dict(
        batch_size=batch_size,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    if dataset.surprise_weights is not None:
        weights = dataset.surprise_weights + 1e-6
        sampler = WeightedRandomSampler(weights, len(dataset), replacement=True)
        loader = DataLoader(dataset, sampler=sampler, **_common)
    else:
        loader = DataLoader(dataset, shuffle=True, **_common)
    return dataset, loader


def _train_one_cycle(model, optimizer, dataloader, device, train_epochs, scheduler=None):
    """Run `train_epochs` epochs over `dataloader`.

    Returns (total, policy_hard, value, soft_policy, num_batches). All four
    loss terms are tracked separately so the dashboard can show hard vs soft
    policy CE without conflating them.
    """
    from training.train import compute_loss

    model.train()
    total_loss_sum = 0.0
    policy_loss_sum = 0.0
    value_loss_sum = 0.0
    soft_policy_loss_sum = 0.0
    mlh_loss_sum = 0.0
    num_batches = 0

    use_amp = (device == 'cuda')
    scaler = torch.amp.GradScaler(enabled=use_amp)

    nb = (device == 'cuda')  # non_blocking only meaningful when pin_memory=True

    for _ in range(train_epochs):
        for batch in dataloader:
            planes = batch[0].to(device, non_blocking=nb)
            policy_target = batch[1].to(device, non_blocking=nb)
            value_target = batch[2].to(device, non_blocking=nb)
            mlh_target = None
            policy_mask = None
            bi = 3
            if bi < len(batch) and batch[bi].dtype in (torch.float32, torch.float64):
                mlh_target = batch[bi].to(device, non_blocking=nb)
                bi += 1
            if bi < len(batch) and batch[bi].dtype == torch.bool:
                policy_mask = batch[bi].to(device, non_blocking=nb)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device, enabled=use_amp):
                policy_logits, value_logits, mlh_pred = model(planes)
                total_loss, policy_loss, value_loss, soft_policy_loss, mlh_loss = compute_loss(
                    policy_logits, value_logits, policy_target, value_target,
                    mlh_pred, mlh_target, policy_mask,
                )
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            # AMP scaler may skip optimizer.step() on inf/nan gradients
            # (common with FP16 on the first few batches). When skipped, the
            # scaler reduces its scale by backoff_factor, so a drop in scale
            # is the canonical signal that the step did NOT happen.
            prev_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None and scaler.get_scale() >= prev_scale:
                scheduler.step()

            total_loss_sum += total_loss.item()
            policy_loss_sum += policy_loss.item()
            value_loss_sum += value_loss.item()
            soft_policy_loss_sum += soft_policy_loss.item()
            mlh_loss_sum += mlh_loss.item()
            num_batches += 1

    return (
        total_loss_sum, policy_loss_sum, value_loss_sum,
        soft_policy_loss_sum, mlh_loss_sum, num_batches,
    )


def _derive_adjudication_rate(shard_path: str) -> float:
    """Fraction of games in an aggregate v2 shard that were adjudicated.

    Uses `game_lengths` to isolate the first row of each game (stamped rows
    share the same `adjudicated` value). Returns 0.0 for pre-v2 shards.
    """
    try:
        data = np.load(shard_path, allow_pickle=False)
    except Exception:
        return 0.0
    if "game_lengths" not in data.files or "adjudicated" not in data.files:
        return 0.0
    lengths = data["game_lengths"]
    adj = data["adjudicated"]
    if len(lengths) == 0:
        return 0.0
    adjud = 0
    cursor = 0
    for gl in lengths:
        gl = int(gl)
        if gl <= 0:
            continue
        if bool(adj[cursor]):
            adjud += 1
        cursor += gl
    return adjud / float(len(lengths))


def _derive_playthrough_min_evals(shard_path: str) -> list[float]:
    """Scan an aggregate v2 shard and return per-playthrough-game min-W of
    the eventual winner (row-by-row, STM-POV flipped if needed).

    Returns an empty list for pre-v2 shards, shards without playthrough rows,
    shards lacking `game_lengths`, or draws (which carry no winner signal).
    """
    try:
        data = np.load(shard_path, allow_pickle=False)
    except Exception:
        return []
    if "game_lengths" not in data.files or "was_playthrough" not in data.files:
        return []
    if "played_eval" not in data.files or "values" not in data.files:
        return []
    lengths = data["game_lengths"]
    wp = data["was_playthrough"]
    played = data["played_eval"]      # (N, 3) WDL stm-POV
    values = data["values"]           # (N, 3) game result stm-POV
    out: list[float] = []
    cursor = 0
    for gl in lengths:
        gl = int(gl)
        if gl <= 0:
            continue
        end = cursor + gl
        game_wp = wp[cursor:end]
        if not game_wp.any():
            cursor = end
            continue
        # Decide winner from the first row's stamped `values` (STM POV).
        # values = (w, d, l): if w >> l then STM at this row won; if l >> w,
        # opponent of STM at this row won; draws we skip.
        first_v = values[cursor]
        if first_v[1] >= max(first_v[0], first_v[2]) - 1e-6:
            cursor = end
            continue
        stm_first_won = first_v[0] > first_v[2]
        # For each row in the game, compute eventual winner's W from this
        # row's STM POV: if row's STM is the winner's color → played_eval[0];
        # else played_eval[2] (the STM's loss prob = the winner's win prob).
        per_row: list[float] = []
        for i in range(cursor, end):
            row_v = values[i]
            row_stm_is_winner = (row_v[0] > row_v[2]) == stm_first_won
            # Equivalent: the row's STM lines up with the winner iff its own
            # game-result W > L. Simpler/robust: just check row_v directly.
            row_stm_is_winner = row_v[0] > row_v[2]
            w_for_winner = float(played[i][0] if row_stm_is_winner else played[i][2])
            per_row.append(w_for_winner)
        if per_row:
            out.append(min(per_row))
        cursor = end
    return out


def training_loop(
    generations: int = 10,
    games_per_gen: int = 400,
    train_epochs: int = 2,
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
    restore_from_checkpoint: list[str] | None = None,
    lr_milestones: list[int] | None = None,
    lr_gamma: float = 0.1,
    mcts_batch_size: int | None = None,
    continuous_flow: bool = True,
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
    from torch.optim.lr_scheduler import LinearLR, MultiStepLR, SequentialLR

    def _build_warmup_scheduler(
        opt,
        steps: int = 250,
        milestones: list[int] | None = None,
        gamma: float = 0.1,
    ):
        """LinearLR warmup over `steps` batches; caller steps it per batch.

        If `milestones` is given, a MultiStepLR decay is chained after warmup
        via SequentialLR. Milestones are absolute batch-step indices measured
        from scheduler construction. On resume / tier transition the scheduler
        is rebuilt, so milestones restart counting from 0 at that point.
        """
        warmup = LinearLR(opt, start_factor=1e-6, end_factor=1.0, total_iters=steps)
        if not milestones:
            return warmup
        # MultiStepLR milestones are relative to its own step counter, which
        # SequentialLR starts from 0 after the warmup phase ends.
        decay = MultiStepLR(opt, milestones=list(milestones), gamma=gamma)
        return SequentialLR(opt, [warmup, decay], milestones=[steps])

    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # cuDNN autotune: lets the first batch of each shape probe convolution
    # algorithms and cache the fastest. Our tensor shapes are stable across
    # batches (only batch dim varies, and it's pinned by drop_last=True), so
    # the autotune cost amortizes across all subsequent forwards.
    # TF32 matmuls are a free ~2× on Ampere for the FC layers in the heads.
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')

    data_dir = os.path.join(output_dir, 'data')
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    from training.metrics import MetricsLogger, TrainingMetrics
    metrics_dir = os.path.join(output_dir, 'metrics')
    metrics_logger = MetricsLogger(metrics_dir)

    # Auto-resume: if the caller didn't pass --resume-from but the output
    # directory already contains checkpoints, pick up the most recent one.
    # Matches the common case where a run was interrupted and the user just
    # re-runs the same command; avoids accidentally starting over at gen 1
    # against an existing metrics/ dir (which breaks the dashboard counter).
    if resume_from is None:
        try:
            existing = [
                f for f in os.listdir(checkpoint_dir)
                if f.startswith('model_gen_') and f.endswith('.pt')
            ]
            if existing:
                def _gen_of(fname: str) -> int:
                    return int(fname[len('model_gen_'):-len('.pt')])
                latest = max(existing, key=_gen_of)
                resume_from = os.path.join(checkpoint_dir, latest)
                print(f"Auto-resuming from most recent checkpoint: {latest}")
                print(f"  (pass --resume-from explicitly, or use a fresh --output-dir, to override)")
        except FileNotFoundError:
            pass

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
        # Construct the LR warmup scheduler AFTER load_state_dict: LinearLR's
        # initial step multiplies the current param_group['lr'] by start_factor,
        # and later steps derive the next lr from the current lr via a formula
        # that assumes lr was set by the scheduler (not restored from a
        # checkpoint). Reversing the order causes lr to explode by ~1e6×
        # after the first training step (lr=1e-3 → ~4 → ~1000).
        scheduler = _build_warmup_scheduler(optimizer, milestones=lr_milestones, gamma=lr_gamma)
        # Restore scheduler state if present, so warmup + milestones don't
        # restart from step 0 on every resume (which would re-apply the linear
        # warmup and silently shift milestone steps relative to absolute step).
        sched_state = checkpoint.get('scheduler_state_dict')
        if sched_state is not None:
            try:
                scheduler.load_state_dict(sched_state)
            except (ValueError, KeyError) as e:
                print(f"  Scheduler state could not be restored ({e}); fresh schedule.")
        # Restore tier schedule from checkpoint if caller didn't override it.
        # Lets auto-resume preserve the original run's scale-up plan instead
        # of silently dropping it (which would pin the net at the small tier).
        saved_schedule = checkpoint.get('network_schedule')
        if network_schedule is None and saved_schedule:
            network_schedule = [tuple(t) for t in saved_schedule]
            print(f"  Restored network_schedule from checkpoint: {network_schedule}")

        # Restore hyperparameters the caller didn't explicitly set.
        # restore_from_checkpoint names the keys to auto-inherit; everything
        # else is warned about on mismatch (visibility without auto-override).
        saved_params = checkpoint.get('run_params') or {}
        inherit = set(restore_from_checkpoint or ())
        if 'adaptive' in inherit and saved_params.get('adaptive'):
            adaptive = AdaptiveConfig(**saved_params['adaptive'])
            print(f"  Restored adaptive config from checkpoint: enabled={adaptive.enabled}")
        if 'use_trt' in inherit and 'use_trt' in saved_params:
            use_trt = saved_params['use_trt']
            print(f"  Restored use_trt={use_trt} from checkpoint")

        # Warn on silent hyperparameter drift (values the user didn't
        # actively restore but that differ from the saved run).
        mismatches = []
        for key, current in (
            ('games_per_gen', games_per_gen), ('batch_size', batch_size),
            ('lr', lr), ('window_size', window_size),
            ('num_simulations', num_simulations), ('max_moves', max_moves),
            ('parallel_games', parallel_games), ('train_epochs', train_epochs),
        ):
            if key in saved_params and saved_params[key] != current:
                mismatches.append((key, saved_params[key], current))
        if mismatches:
            print("  [warn] hyperparameters differ from saved run (using current):")
            for key, saved, cur in mismatches:
                print(f"    {key}: saved={saved}, current={cur}")

        print(f"  Loaded generation {start_gen - 1} ({blocks}b{filters}f), resuming from gen {start_gen}")
    else:
        # Pick initial tier if schedule is set.
        blocks, filters = resolve_tier(1, network_schedule, blocks, filters)
        config = NetworkConfig(num_blocks=blocks, num_filters=filters)
        model = ChessNetwork(config).to(device)
        optimizer = create_optimizer(model, lr=lr, weight_decay=weight_decay)
        scheduler = _build_warmup_scheduler(optimizer, milestones=lr_milestones, gamma=lr_gamma)

    # Stochastic Weight Averaging: use averaged model for self-play (smoother policy)
    swa_model = None
    if use_swa:
        from torch.optim.swa_utils import AveragedModel
        # Match Lc0's approach: average parameters only. BN running stats come
        # from the current model (or a post-training update_bn() pass).
        # use_buffers=True pulls in int64 num_batches_tracked, which trips
        # PyTorch 2.9's SWA multi_avg_fn on CUDA (_foreach_addcdiv_ integer
        # path is disallowed).
        swa_model = AveragedModel(model)
        # Restore SWA EMA state if resuming, so n_averaged and the running
        # averaged weights survive across restarts. Without this, every resume
        # would silently reset the average to "just the current model".
        if resume_from:
            swa_state = checkpoint.get('swa_state_dict')
            if swa_state is not None:
                try:
                    swa_model.load_state_dict(swa_state)
                    print(f"  Restored SWA state (n_averaged={int(swa_model.n_averaged.item())})")
                except (ValueError, KeyError, RuntimeError) as e:
                    print(f"  SWA state could not be restored ({e}); fresh SWA.")

    # Enable Lc0-style search features for the training loop:
    # - Absolute root FPU: unvisited root children treated as value=1.0 (pessimistic),
    #   encouraging broader early root exploration.
    # - MLH PUCT bonus: in winning/losing positions, prefer children that shorten/lengthen
    #   the game. Disabled until MLH outputs diverge (which requires training signal).
    _mcts_kwargs = dict(
        num_simulations=num_simulations,
        policy_softmax_temperature=1.0,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.15,     # lower noise for pretrained net (was 0.25)
        fpu_absolute_root=True,
        fpu_absolute_root_value=1.0,
        mlh_weight=0.03,
        mlh_cap=8.0,
        mlh_q_threshold=0.6,
    )
    if mcts_batch_size is not None:
        _mcts_kwargs['batch_size'] = mcts_batch_size
    mcts_config = MCTSConfig(**_mcts_kwargs)
    selfplay_config = SelfPlayConfig(
        max_moves=max_moves,
        resign_threshold=resign_threshold,
        syzygy_path=syzygy_path,
        opening_book_path=opening_book_path,
        opening_book_fraction=opening_book_fraction,
    )

    # Initialize the C++ Syzygy tablebase wrapper so MCTS leaves with
    # piece_count <= TB_LARGEST resolve exactly via Fathom and bypass the NN.
    # Without this call, only the post-game Python rescore runs — MCTS still
    # spends GPU time on endgame leaves the tablebase could answer for free.
    if syzygy_path:
        try:
            import chess_mcts
            n = chess_mcts.syzygy_init(syzygy_path)
            print(f"[syzygy] in-search probing enabled (max-pieces={n})")
        except Exception as e:
            print(f"[syzygy] in-search init failed ({e}); using Python rescore only")

    if opening_book_path:
        fens = _load_opening_book(opening_book_path)
        print(f"[opening-book] loaded {len(fens)} FENs from {opening_book_path}")

    # Default adaptive config if not provided
    if adaptive is None:
        adaptive = AdaptiveConfig(enabled=False, full_sims=num_simulations,
                                   full_max_moves=max_moves, full_games=games_per_gen)

    # Stage 6: discard pool — positions rejected by the min-visit floor are
    # pushed here and can seed future games for diversity.
    from training.discard_pool import DiscardPool
    discard_pool = DiscardPool(cap=10000, persist_path=os.path.join(output_dir, 'discard_pool.json'))

    # Stage 7: EMA-auto-tune the resign threshold from playthrough shards.
    # `selfplay_config.resign_w` is overwritten after each generation.
    from training.resign_calibrator import ResignCalibrator
    resign_calibrator = ResignCalibrator(
        default=selfplay_config.resign_w,
        warmup_generations=3,
    )

    end_gen = start_gen + generations - 1
    print(f"Starting training loop: generations {start_gen}-{end_gen}, {games_per_gen} games/gen")
    print(f"Device: {device}, Model: {blocks} blocks, {filters} filters"
          f"{', SWA enabled' if use_swa else ''}"
          f"{', adaptive' if adaptive.enabled else ''}"
          f"{f', tiered schedule={network_schedule}' if network_schedule else ''}")

    # Used to mark the first gen after a resume on the dashboard's loss chart.
    first_gen_after_resume = start_gen if resume_from else None

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
            scheduler = _build_warmup_scheduler(optimizer, milestones=lr_milestones, gamma=lr_gamma)
            if use_swa:
                from torch.optim.swa_utils import AveragedModel
                swa_model = AveragedModel(model)
            blocks, filters = target_blocks, target_filters

            # Warm-up: train the new (random) net on prior-gen data before
            # self-play, so this generation's games aren't played by a fresh
            # random network. Skipped when there's no prior data (gen==1).
            warmup_start = max(1, gen - window_size + 1)
            warmup_end = gen - 1
            if warmup_end >= warmup_start:
                _, warmup_loader = _build_window_dataloader(
                    data_dir, warmup_start, warmup_end, batch_size,
                )
                if warmup_loader is not None:
                    print(f"  Warm-up training new net on window {warmup_start}-{warmup_end}"
                          f" ({train_epochs} epochs) before self-play")
                    _train_one_cycle(model, optimizer, warmup_loader, device, train_epochs, scheduler)
                    if swa_model is not None:
                        swa_model.update_parameters(model)

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
                trt_engine_path = _build_trt_engine_for_self_play(
                    model, cpp_model_path, max_batch=max(512, mcts_config.batch_size),
                )
            model.load_state_dict(orig_state)
        else:
            export_torchscript(model, cpp_model_path, device=device)
            if use_trt:
                trt_engine_path = _build_trt_engine_for_self_play(
                    model, cpp_model_path, max_batch=max(512, mcts_config.batch_size),
                )

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
            discard_pool=discard_pool,
            continuous_flow=continuous_flow,
        )

        # Stage 7: EMA-update resign_w from this generation's playthrough
        # games. During warmup the calibrator leaves resign_w at its default.
        pt_samples = _derive_playthrough_min_evals(data_path)
        if pt_samples:
            resign_calibrator.false_positive_rate(pt_samples)
        new_resign_w = resign_calibrator.update(generation=gen, playthrough_min_evals=pt_samples)
        selfplay_config.resign_w = new_resign_w

        # 2. Load sliding window of recent generations
        window_start = max(1, gen - window_size + 1)
        dataset, dataloader = _build_window_dataloader(
            data_dir, window_start, gen, batch_size,
        )
        print(f"Training on {len(dataset)} samples "
              f"(window {window_start}-{gen}, {train_epochs} epoch(s))...", flush=True)

        # 3. Train for train_epochs epochs (with mixed precision on CUDA)
        (total_loss_sum, policy_loss_sum, value_loss_sum,
         soft_policy_loss_sum, mlh_loss_sum, num_batches) = _train_one_cycle(
            model, optimizer, dataloader, device, train_epochs, scheduler,
        )

        # Update SWA model after training (parameters only, Lc0-style).
        if swa_model is not None:
            swa_model.update_parameters(model)

        # 4. Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'model_gen_{gen}.pt')
        run_params = {
            'games_per_gen': games_per_gen,
            'train_epochs': train_epochs,
            'batch_size': batch_size,
            'lr': lr,
            'weight_decay': weight_decay,
            'num_simulations': num_simulations,
            'window_size': window_size,
            'max_moves': max_moves,
            'resign_threshold': resign_threshold,
            'use_swa': use_swa,
            'syzygy_path': syzygy_path,
            'opening_book_path': opening_book_path,
            'opening_book_fraction': opening_book_fraction,
            'adaptive': asdict(adaptive) if adaptive else None,
            'parallel_games': parallel_games,
            'use_trt': use_trt,
        }
        ckpt_payload = {
            'generation': gen,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'config': config,
            'network_schedule': network_schedule,
            'run_params': run_params,
        }
        if swa_model is not None:
            ckpt_payload['swa_state_dict'] = swa_model.state_dict()
            ckpt_payload['swa_n_averaged'] = int(swa_model.n_averaged.item())
        torch.save(ckpt_payload, checkpoint_path)

        # 5. Print per-generation stats
        gen_time = time.time() - gen_start
        avg_total = total_loss_sum / max(num_batches, 1)
        avg_policy = policy_loss_sum / max(num_batches, 1)
        avg_value = value_loss_sum / max(num_batches, 1)
        avg_soft = soft_policy_loss_sum / max(num_batches, 1)
        avg_mlh = mlh_loss_sum / max(num_batches, 1)
        print(f"Gen {gen} complete: {num_positions} positions, "
              f"{len(dataset)} training samples (window {window_start}-{gen})")
        print(f"  Loss: total={avg_total:.4f}, policy={avg_policy:.4f},"
              f" value={avg_value:.4f}, soft_policy={avg_soft:.4f},"
              f" mlh={avg_mlh:.4f}")
        print(f"  Time: {gen_time:.1f}s, Checkpoint: {checkpoint_path}")
        training_metrics = TrainingMetrics(
            total_loss=avg_total,
            policy_loss=avg_policy,
            value_loss=avg_value,
            num_batches=num_batches,
            learning_rate=optimizer.param_groups[0]['lr'],
            soft_policy_loss=avg_soft,
            mlh_loss=avg_mlh,
        )
        # Stage 10: operational metrics for the dashboard. Pool size reads 0
        # until a training_loop-owned DiscardPool is wired (future work);
        # FP-rate and adjudication-rate come from this gen's shard.
        adjud_rate = _derive_adjudication_rate(data_path)
        fp_rate = (
            resign_calibrator.last_fp_rate
            if resign_calibrator.last_fp_rate is not None
            else 0.0
        )
        pool_size = discard_pool.size() if discard_pool is not None else 0
        metrics_logger.save_generation(
            generation=gen,
            num_positions=num_positions,
            training=training_metrics,
            duration_s=gen_time,
            network={'blocks': config.num_blocks, 'filters': config.num_filters},
            resumed=(gen == first_gen_after_resume),
            resign_w=float(resign_calibrator.current),
            resign_fp_rate=float(fp_rate),
            discard_pool_size=int(pool_size),
            adjudication_rate=float(adjud_rate),
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
    loop_parser.add_argument('--generations', type=int, default=100)
    loop_parser.add_argument('--games-per-gen', type=int, default=512)
    loop_parser.add_argument('--train-epochs', type=int, default=1)
    loop_parser.add_argument('--batch-size', type=int, default=2048)
    loop_parser.add_argument('--lr', type=float, default=2e-3)
    loop_parser.add_argument('--weight-decay', type=float, default=1e-4)
    loop_parser.add_argument('--simulations', type=int, default=600)
    loop_parser.add_argument('--blocks', type=int, default=10)
    loop_parser.add_argument('--filters', type=int, default=128)
    loop_parser.add_argument('--window-size', type=int, default=5)
    loop_parser.add_argument('--output-dir', type=str, default='models/current_run')
    loop_parser.add_argument('--device', type=str, default='auto')
    loop_parser.add_argument('--max-moves', type=int, default=512)
    loop_parser.add_argument('--resign-threshold', type=float, default=-0.95,
                             help='Legacy scalar resign threshold (Python MCTS fallback path)')
    loop_parser.add_argument('--syzygy', type=str, default=None, help='Path to Syzygy tablebase files')
    loop_parser.add_argument('--opening-book', type=str, default=None, help='Path to opening book FEN file (one FEN per line)')
    loop_parser.add_argument('--opening-book-fraction', type=float, default=0.5, help='Fraction of games seeded from book')
    loop_parser.add_argument('--adaptive', action='store_true', default=argparse.SUPPRESS, help='Enable adaptive settings per generation')
    loop_parser.add_argument('--no-adaptive', dest='adaptive', action='store_false', default=argparse.SUPPRESS)
    loop_parser.add_argument('--early-sims', type=int, default=400, help='Simulations for early generations')
    loop_parser.add_argument('--early-max-moves', type=int, default=300, help='Max moves for early generations')
    loop_parser.add_argument('--early-games', type=int, default=512, help='Games per early generation')
    loop_parser.add_argument('--resume-from', type=str, default=None, help='Resume from a checkpoint file (e.g. selfplay_output/checkpoints/model_gen_5.pt)')
    loop_parser.add_argument('--parallel-games', type=int, default=128, help='Concurrent self-play games (C++ GameManager cross-game batching)')
    loop_parser.add_argument('--mcts-batch-size', type=int, default=256,
                             help='MCTS leaves gathered per GPU forward pass during self-play. '
                                  '256 is tuned for 128 parallel games (~2 leaves/game/pass). '
                                  'TRT engine max_batch auto-scales to match.')
    loop_parser.add_argument(
        '--continuous-flow', dest='continuous_flow', action='store_true',
        default=True,
        help='Continuous-flow self-play: slots respawn as games finish, '
             'eliminating batch-boundary tail latency. (default: on)',
    )
    loop_parser.add_argument(
        '--no-continuous-flow', dest='continuous_flow', action='store_false',
        help='Legacy batch-boundary mode: wait for all slots to finish '
             'before starting the next batch of games.',
    )
    loop_parser.add_argument('--use-trt', dest='use_trt', action='store_true', default=argparse.SUPPRESS, help='Export ONNX + build TensorRT engine per generation and run self-play through the TRT backend (default: on)')
    loop_parser.add_argument('--no-use-trt', dest='use_trt', action='store_false', default=argparse.SUPPRESS, help='Disable TensorRT backend (use LibTorch only)')
    loop_parser.add_argument('--network-schedule', type=str, default=None,
                             help='Tiered net schedule, e.g. "1:6:64,20:10:128" (gen_start:blocks:filters, comma-separated)')
    loop_parser.add_argument('--lr-milestones', type=str, default=None,
                             help='Comma-separated batch-step indices for MultiStepLR decay after warmup, e.g. "30000,60000"')
    loop_parser.add_argument('--lr-gamma', type=float, default=0.1,
                             help='LR multiplier applied at each --lr-milestones step (default 0.1)')

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
        schedule = None
        raw = getattr(args, 'network_schedule', None)
        if raw:
            schedule = []
            for tier in raw.split(','):
                parts = tier.strip().split(':')
                if len(parts) != 3:
                    parser.error(f"--network-schedule tier '{tier}' must be gen_start:blocks:filters")
                schedule.append((int(parts[0]), int(parts[1]), int(parts[2])))

        # argparse.SUPPRESS means the flag only appears on the Namespace when
        # the user passed it. Tracking explicitness lets us auto-restore from
        # the checkpoint's run_params when the user didn't override.
        adaptive_explicit = hasattr(args, 'adaptive')
        use_trt_explicit = hasattr(args, 'use_trt')

        adaptive_config = None
        if adaptive_explicit and args.adaptive:
            adaptive_config = AdaptiveConfig(
                enabled=True,
                early_sims=getattr(args, 'early_sims', 100),
                early_max_moves=getattr(args, 'early_max_moves', 150),
                early_games=getattr(args, 'early_games', 300),
                full_sims=args.simulations,
                full_max_moves=args.max_moves,
                full_games=args.games_per_gen,
            )

        restore = []
        if not adaptive_explicit:
            restore.append('adaptive')
        if not use_trt_explicit:
            restore.append('use_trt')

        lr_milestones = None
        if getattr(args, 'lr_milestones', None):
            lr_milestones = [int(x) for x in args.lr_milestones.split(',') if x.strip()]

        training_loop(
            generations=args.generations,
            games_per_gen=args.games_per_gen,
            train_epochs=args.train_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=getattr(args, 'weight_decay', 1e-4),
            num_simulations=args.simulations,
            blocks=args.blocks,
            filters=args.filters,
            window_size=args.window_size,
            output_dir=args.output_dir,
            device=args.device,
            max_moves=args.max_moves,
            resign_threshold=getattr(args, 'resign_threshold', -0.95),
            syzygy_path=getattr(args, 'syzygy', None),
            opening_book_path=getattr(args, 'opening_book', None),
            opening_book_fraction=getattr(args, 'opening_book_fraction', 0.5),
            adaptive=adaptive_config,
            resume_from=getattr(args, 'resume_from', None),
            parallel_games=getattr(args, 'parallel_games', 128),
            use_trt=getattr(args, 'use_trt', True),
            network_schedule=schedule,
            restore_from_checkpoint=restore,
            lr_milestones=lr_milestones,
            lr_gamma=getattr(args, 'lr_gamma', 0.1),
            mcts_batch_size=getattr(args, 'mcts_batch_size', None),
            continuous_flow=getattr(args, 'continuous_flow', True),
        )


if __name__ == '__main__':
    main()
