"""GamePoolManager: per-step Python orchestration for self-play.

Owns game lifecycle. C++ GameManager is a pure search engine that exposes
RootStats; this module decides what move to play, when to resign, when to
adjudicate, and what to record.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

# Standard start position. Used by `_respawn_slot` to launch a fresh game
# into a finished slot. If we ever diversify launch positions (opening
# books, curriculum FENs), this constant moves behind a config hook that
# `init_games_from_fen` and `_respawn_slot` both read.
_STARTPOS_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


@dataclass
class StepRow:
    """One position's training row (pre-final-result-stamp)."""
    fen: str
    visits_policy: list  # visit distribution over legal moves (normalized)
    soft_policy: list    # (1858,) raw NN policy
    best_eval: tuple     # (w,d,l) at argmax-visit child
    played_eval: tuple   # (w,d,l) at played child
    raw_nn_eval: tuple   # (w,d,l) raw NN at this position
    mlh: float
    side_to_move: int    # 0=white, 1=black
    is_full_search: bool
    was_playthrough: bool
    adjudicated: bool = False
    # Auxiliary fields needed by the Stage 2 play_games_batched adapter so it
    # can project `visits_policy` back into the 1858-dim index space used by
    # the current on-disk format. Not consumed by Stage 3 writers.
    legal_moves_uci: list = field(default_factory=list)
    n_legal: int = 0
    # UCI of the move actually played from this position. Used by the legacy
    # adapter to populate GameRecord.moves_uci (dashboard replay).
    played_uci: str = ""


@dataclass
class GameRecord:
    rows: list = field(default_factory=list)
    seed_source: str = "standard"
    final_wdl: tuple = (0.0, 0.0, 0.0)  # stamped at game end
    adjudicated: bool = False
    was_playthrough: bool = False
    # Terminal status from the C++ side (+1 = STM loses, -1 = STM wins, 2 = draw,
    # 0 = still running when finalize was called without a real terminal).
    terminal_status: int = 0


def _temperature_sample(visits, tau, rng):
    """Multinomial over visits with softmax temperature tau."""
    if tau < 1e-3:
        return max(range(len(visits)), key=lambda i: visits[i])
    counts = [v + 1e-8 for v in visits]
    total = sum(counts)
    weights = [(c / total) ** (1.0 / tau) for c in counts]
    s = sum(weights)
    r = rng.random() * s
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if acc >= r:
            return i
    return len(weights) - 1


def _apply_uci_to_fen(fen, uci_move):
    """Return the FEN that results from playing `uci_move` in `fen`, or None.

    Rejects positions that are terminal or near-terminal (fewer than 5 legal
    moves) so the discard pool doesn't seed games that end immediately.

    Tolerant of malformed inputs — returns None rather than raising so the
    discard-pool push is best-effort and never breaks the self-play loop.
    """
    try:
        import chess  # lazy to keep module import cheap for unit tests
    except Exception:
        return None
    try:
        b = chess.Board(fen)
        b.push_uci(uci_move)
        if b.is_game_over(claim_draw=True):
            return None
        # Reject near-terminal positions — too few legal moves to produce a
        # meaningful game.
        if len(list(b.legal_moves)) < 5:
            return None
        return b.fen()
    except Exception:
        return None


def _random_opening_moves(max_moves, rng):
    """Generate a short sequence of random legal moves from startpos."""
    import chess
    board = chess.Board()
    moves = []
    n = rng.randint(1, max_moves)
    for _ in range(n):
        legal = list(board.legal_moves)
        if not legal:
            break
        m = rng.choice(legal)
        moves.append(m.uci())
        board.push(m)
    return moves


def _resign_check(cfg, root_wdl, ply, is_playthrough):
    """Decide whether to resign based on root WDL.

    Playthrough games (set per-game) suppress resign entirely, so the resulting
    shards let the calibrator measure how often we *would* have resigned on a
    win/draw. We also gate resign until `resign_earliest_ply` to avoid firing
    on noisy opening evals.

    Returns (triggered, outcome) where outcome is "loss_for_stm" | "draw" | None.
    """
    if is_playthrough or ply < cfg.resign_earliest_ply:
        return False, None
    w, d, l = root_wdl
    if w < cfg.resign_w:
        return True, "loss_for_stm"
    if d > cfg.resign_d:
        return True, "draw"
    if l > cfg.resign_l:
        return True, "loss_for_stm"
    return False, None


def _kld(raw_policy_indices, visit_dist):
    """KL(raw || visits) over the legal moves only."""
    eps = 1e-8
    total = sum(visit_dist) + eps
    kld = 0.0
    for raw_p, v in zip(raw_policy_indices, visit_dist):
        q = (v + eps) / total
        if raw_p > eps:
            kld += raw_p * math.log(raw_p / q)
    return max(0.0, kld)


class GamePoolManager:
    """Per-step Python orchestration for self-play.

    Supports two execution modes on the same underlying C++ GameManager:
      - run_until_all_complete(): fixed pool of N slots, runs each slot's
        game to completion once (batch-boundary mode, original behavior).
      - run_pool(target_games): continuous flow — as each slot's game ends,
        immediately respawn a fresh game into it until `target_games` total
        have completed. Used by play_games_batched to eliminate tail-latency
        waste at batch boundaries.
    """

    def __init__(self, game_manager, cfg, discard_pool=None, rng_seed=None,
                 opening_book_fens=None):
        self.gm = game_manager
        self.cfg = cfg
        self.discard_pool = discard_pool
        self.opening_book_fens = opening_book_fens or []
        self.rng = random.Random(rng_seed)
        self.n = game_manager.num_games()
        self._records = [GameRecord() for _ in range(self.n)]
        # Stage 7: flag a fraction of games as playthroughs — resign is
        # suppressed so the calibrator can observe the true distribution of
        # eventual-winner min-W. Read defensively: callers may supply a cfg
        # object that predates this field.
        pt_frac = float(getattr(cfg, "resign_playthrough_fraction", 0.0) or 0.0)
        for rec in self._records:
            rec.was_playthrough = self.rng.random() < pt_frac if pt_frac > 0 else False
        self._target_sims = [cfg.full_sims] * self.n
        self._last_kld = [0.0] * self.n
        self._completed = [False] * self.n
        # Continuous-flow state (used only by run_pool; run_until_all_complete
        # ignores these fields). Initialized here so tests that poke state
        # directly can rely on them existing.
        self._completed_games: list[GameRecord] = []
        self._launched: int = self.n       # slots initially launched by init_games()

    def _next_target_sims(self, game_idx, last_was_full):
        """Playout-cap randomization + KLD-adaptive interpolation."""
        cfg = self.cfg
        if self.rng.random() < cfg.playout_cap_p:
            return cfg.quick_sims
        if cfg.use_kld_adaptive and last_was_full:
            kld = self._last_kld[game_idx]
            ratio = max(0.0, min(1.0, kld / cfg.kld_threshold))
            max_sims = getattr(cfg, 'kld_max_sims', cfg.full_sims)
            return int(cfg.min_sims + ratio * (max_sims - cfg.min_sims))
        return cfg.full_sims

    def _min_visit_floor(self, i):
        """Resolve the visit-count floor for game `i`.

        None (the default) means auto-scale to max(5, 1% of target_sims), which
        matches Lc0's heuristic. A concrete int is an absolute threshold.
        """
        explicit = getattr(self.cfg, "min_visits_floor", None)
        if explicit is not None:
            return int(explicit)
        return max(5, int(self._target_sims[i] * 0.01))

    def _temperature(self, ply):
        cfg = self.cfg
        if ply < cfg.opening_temp_plies:
            return cfg.opening_temp
        # Linear decay to floor.
        decay = (ply - cfg.opening_temp_plies) / max(1, cfg.temp_decay_plies)
        return max(cfg.temp_floor, cfg.opening_temp * (1.0 - decay))

    def step_once(self):
        """One cross-game batched step.

        Returns the number of games that made progress (either advanced a
        move or completed). Used by run_until_all_complete and run_pool to
        detect real stagnation. Slots flagged `_completed[i]=True` get
        target_sims=0 so C++ skips them entirely.
        """
        targets = [
            0 if self._completed[i] else self._target_sims[i]
            for i in range(self.n)
        ]
        stats = self.gm.step_stats(targets)
        advanced = 0
        for i, s in enumerate(stats):
            if self._completed[i]:
                continue
            if s.terminal_status != 0:
                self._finalize_game(i, terminal_status=s.terminal_status, adjudicated=False)
                advanced += 1
                continue
            self._record_and_play(i, s)
            advanced += 1
        return advanced

    def _record_and_play(self, i, s):
        cfg = self.cfg
        rec = self._records[i]
        ply = self.gm.get_ply(i)
        tau = self._temperature(ply)
        was_full_search = (self._target_sims[i] >= cfg.full_sims)
        # Move selection with min-visit floor. Positions the temperature
        # sampler picks but that didn't receive enough search are likely to
        # be policy outliers — reject them, push the would-be resulting
        # position to the discard pool for later use as a starting FEN, and
        # resample. Fall back to argmax after 3 retries.
        chosen = _temperature_sample(s.visits, tau, self.rng)
        floor = self._min_visit_floor(i)
        if floor > 0 and self.discard_pool is not None:
            visits_work = list(s.visits)
            legal = list(getattr(s, "legal_moves_uci", []) or [])
            retries = 0
            while (chosen != s.best_child_idx
                   and 0 <= chosen < len(visits_work)
                   and visits_work[chosen] < floor
                   and retries < 3):
                if 0 <= chosen < len(legal):
                    cand_fen = _apply_uci_to_fen(self.gm.get_fen(i), legal[chosen])
                    if cand_fen is not None:
                        self.discard_pool.push(cand_fen)
                visits_work[chosen] = 0
                chosen = _temperature_sample(visits_work, tau, self.rng)
                retries += 1
            if (0 <= chosen < len(visits_work)
                    and visits_work[chosen] < floor
                    and chosen != s.best_child_idx):
                chosen = s.best_child_idx

        # Build rows (Stage 3 will swap to v2 schema).
        played_q = s.q_per_child[chosen] if 0 <= chosen < len(s.q_per_child) else 0.0
        best_q = s.q_per_child[s.best_child_idx] if 0 <= s.best_child_idx < len(s.q_per_child) else 0.0

        # Pull legal_moves_uci — available as a property on the C++ RootStats
        # binding and on the duck-typed fakes used in tests.
        legal_moves_uci = list(getattr(s, "legal_moves_uci", []) or [])

        played_uci = legal_moves_uci[chosen] if 0 <= chosen < len(legal_moves_uci) else ""
        row = StepRow(
            fen=self.gm.get_fen(i),
            visits_policy=_normalize(s.visits),
            soft_policy=list(s.raw_nn_policy),
            best_eval=_q_to_wdl(best_q),
            played_eval=_q_to_wdl(played_q),
            raw_nn_eval=tuple(s.raw_nn_value),
            mlh=s.raw_nn_mlh,
            side_to_move=ply % 2,
            is_full_search=was_full_search,
            was_playthrough=rec.was_playthrough,
            legal_moves_uci=legal_moves_uci,
            n_legal=s.n_legal,
            played_uci=played_uci,
        )
        rec.rows.append(row)

        # Update KLD for next-step target.
        if was_full_search:
            self._last_kld[i] = _kld(s.raw_nn_policy[: s.n_legal], s.visits)

        # Apply move.
        try:
            self.gm.apply_move(i, chosen)
        except RuntimeError as e:
            raise RuntimeError(
                f"apply_move failed on slot {i}: chosen={chosen}, "
                f"best_child_idx={s.best_child_idx}, n_legal={s.n_legal}, "
                f"visits_len={len(s.visits)}, terminal_status={s.terminal_status}, "
                f"ply={ply}, tau={tau:.3f}, full_sims={cfg.full_sims}, "
                f"target_sims[i]={self._target_sims[i]}, "
                f"sims_done={getattr(s, 'sims_done', '?')}, "
                f"visits_sum={sum(s.visits) if s.visits else 0}, "
                f"visits_head={list(s.visits)[:5]}, "
                f"legal_head={list(getattr(s, 'legal_moves_uci', []))[:5]}, "
                f"fen={self.gm.get_fen(i)!r}"
            ) from e

        # WDL-aware resign. Stage 4: uses s.root_wdl from the pre-apply stats
        # (the eval that informed the just-played move), gated by
        # resign_earliest_ply and suppressed for playthrough games.
        triggered, outcome = _resign_check(
            cfg, tuple(s.root_wdl), ply, rec.was_playthrough,
        )
        if triggered:
            ts = 2 if outcome == "draw" else -1
            self._finalize_game(i, terminal_status=ts, adjudicated=False)
            return

        # Ply-cap adjudication.
        if self.gm.get_ply(i) >= cfg.max_ply:
            self._finalize_game(i, terminal_status=2, adjudicated=True)
            return

        # Compute next-step target.
        self._target_sims[i] = self._next_target_sims(i, was_full_search)

    def _respawn_slot(self, i: int) -> None:
        """Reset per-slot state and kick off a fresh game in C++ slot i.

        Called from run_pool's finalize path after the slot's completed
        GameRecord has been collected into `self._completed_games`.
        Re-rolls the playthrough flag; resets target_sims, last_kld,
        completed flag, and the per-slot GameRecord. Uses
        chess_mcts.GameManager.init_game_from_fen which allocates a fresh
        root Node for this slot without resetting the shared NodePool.

        Starting FEN selection priority:
          1. Discard pool (cfg.discarded_start_chance)
          2. Opening book (if provided)
          3. Random opening (cfg.random_opening_fraction)
          4. Standard startpos
        """
        cfg = self.cfg
        self._records[i] = GameRecord()
        self._target_sims[i] = cfg.full_sims
        self._last_kld[i] = 0.0
        self._completed[i] = False
        pt_frac = float(getattr(cfg, "resign_playthrough_fraction", 0.0) or 0.0)
        self._records[i].was_playthrough = (
            self.rng.random() < pt_frac if pt_frac > 0 else False
        )
        # Pick starting FEN: discard pool → opening book → random opening → startpos
        fen = _STARTPOS_FEN
        moves: list[str] = []
        disc_chance = float(getattr(cfg, "discarded_start_chance", 0.0) or 0.0)
        if self.discard_pool is not None and disc_chance > 0 and self.rng.random() < disc_chance:
            min_pieces = int(getattr(cfg, "discarded_min_pieces", 16))
            for _ in range(3):
                candidate = self.discard_pool.pop()
                if candidate is None:
                    break
                piece_count = sum(1 for ch in candidate.split()[0] if ch.isalpha())
                if piece_count >= min_pieces:
                    fen = candidate
                    break
        if fen == _STARTPOS_FEN and self.opening_book_fens:
            fen = self.rng.choice(self.opening_book_fens)
        rand_frac = float(getattr(cfg, "random_opening_fraction", 0.0) or 0.0)
        if fen == _STARTPOS_FEN and rand_frac > 0 and self.rng.random() < rand_frac:
            rand_moves = int(getattr(cfg, "random_opening_moves", 8))
            moves = _random_opening_moves(rand_moves, self.rng)
        self.gm.init_game_from_fen(i, fen, moves, cfg.full_sims)

    def _finalize_game(self, i, terminal_status, adjudicated):
        rec = self._records[i]
        rec.adjudicated = adjudicated
        rec.terminal_status = terminal_status
        if adjudicated:
            # Ply-cap adjudication: stamp every row so the trainer can
            # downweight them via adjudicated_weight. Decide outcome from the
            # last raw_nn_eval — if confidently one-sided, call it; otherwise
            # treat as a draw.
            for row in rec.rows:
                row.adjudicated = True
            last_wdl = rec.rows[-1].raw_nn_eval if rec.rows else (0.0, 1.0, 0.0)
            if max(last_wdl) > 0.6:
                winner = max(range(3), key=lambda k: last_wdl[k])
                if winner == 0:
                    rec.final_wdl = (1.0, 0.0, 0.0)
                elif winner == 1:
                    rec.final_wdl = (0.0, 1.0, 0.0)
                else:
                    rec.final_wdl = (0.0, 0.0, 1.0)
            else:
                rec.final_wdl = (0.0, 1.0, 0.0)
        else:
            # Real terminal. Convention: +1 = STM loses, -1 = STM wins.
            if terminal_status == 2:
                rec.final_wdl = (0.0, 1.0, 0.0)
            elif terminal_status == 1:
                # STM at terminal loses.
                rec.final_wdl = (0.0, 0.0, 1.0)
            else:
                # STM at terminal wins (or resign: resigning side loses).
                rec.final_wdl = (1.0, 0.0, 0.0)
        self._completed[i] = True

    def run_until_all_complete(self):
        while not all(self._completed):
            advanced = self.step_once()
            if advanced == 0 and not all(self._completed):
                # Safety: avoid infinite loop in pathological cases.
                for i in range(self.n):
                    if not self._completed[i]:
                        self._finalize_game(i, terminal_status=2, adjudicated=True)
        return self._records

    def run_pool(self, target_games: int, on_game_done=None):
        """Continuous-flow self-play: run until `target_games` games have completed.

        As each slot's game reaches a terminal state (natural, resign, or
        ply-cap adjudication), its GameRecord is collected and the slot is
        either respawned (if more games are still needed) or marked inactive.
        Inactive slots get target_sims=0 on subsequent step_stats calls, so
        the C++ GameManager performs no work on them.

        Args:
            target_games: total completed games to produce.
            on_game_done: optional callable (record, completion_idx) fired
                per completed game in completion order.

        Returns:
            list[GameRecord] of exactly `target_games` finished games.
        """
        if target_games <= 0:
            return []
        # Cap launched at target_games. __init__ set _launched = self.n
        # because init_games() kicks off one game per slot — but if the pool
        # is wider than the target (n > target_games), the extras are surplus
        # and should be inactive from step 1 so step_stats doesn't burn GPU
        # cycles on them. Production usage (play_games_batched) caps
        # parallel_games at num_games before constructing the pool, so this
        # branch is defensive; unit tests exercise it directly.
        if self._launched > target_games:
            for i in range(target_games, self._launched):
                self._completed[i] = True
            self._launched = target_games
        max_steps = target_games * max(1, self.cfg.max_ply) + self.n
        step_count = 0
        while len(self._completed_games) < target_games:
            if step_count >= max_steps:
                for i in range(self.n):
                    if not self._completed[i] and len(self._completed_games) < target_games:
                        self._finalize_game(i, terminal_status=2, adjudicated=True)
                        self._harvest_slot(i, on_game_done, target_games)
                break
            step_count += 1
            advanced = self._step_and_harvest(on_game_done, target_games)
            if advanced == 0 and len(self._completed_games) < target_games:
                # Stagnation: no slot advanced and no slot finalized. Kill
                # one active slot as a draw and let the next step_stats run
                # — if other slots were merely slow, they'll catch up. We
                # only adjudicate one slot per tick (not all) so a transient
                # stall doesn't cascade into a bulk-draw wipeout.
                for i in range(self.n):
                    if not self._completed[i]:
                        self._finalize_game(i, terminal_status=2, adjudicated=True)
                        self._harvest_slot(i, on_game_done, target_games)
                        break
        return list(self._completed_games)

    def _step_and_harvest(self, on_game_done, target_games):
        """One step_stats call + harvest/respawn of any completed slots.

        Returns the number of slots that advanced (same semantics as
        step_once) so run_pool can detect stagnation.
        """
        targets = [
            0 if self._completed[i] else self._target_sims[i]
            for i in range(self.n)
        ]
        stats = self.gm.step_stats(targets)
        advanced = 0
        for i, s in enumerate(stats):
            if self._completed[i]:
                continue
            if s.terminal_status != 0:
                self._finalize_game(i, terminal_status=s.terminal_status, adjudicated=False)
                self._harvest_slot(i, on_game_done, target_games)
                advanced += 1
                continue
            self._record_and_play(i, s)
            advanced += 1
            if self._completed[i]:
                self._harvest_slot(i, on_game_done, target_games)
        return advanced

    def _harvest_slot(self, i, on_game_done, target_games):
        """Collect slot i's completed record, then either respawn or mark inactive."""
        record = self._records[i]
        self._completed_games.append(record)
        if on_game_done is not None:
            on_game_done(record, len(self._completed_games))
        # Two guards, both necessary:
        #   _launched < target_games     — don't start more games than asked for
        #   len(completed) < target_games — don't respawn on the very last completion
        # (that slot is about to exit inactive; respawning would race the loop-exit
        # check in run_pool).
        if self._launched < target_games and len(self._completed_games) < target_games:
            self._respawn_slot(i)
            self._launched += 1


def _normalize(visits):
    s = sum(visits) or 1
    return [v / s for v in visits]


# TODO(stage 3): replace with per-child WDL plumbed through RootStats. For
# now, approximate WDL from scalar Q in [-1, 1]. This preserves Stage 2's
# behavior-preserving contract and will be swapped for true WDL in the
# schema-v2 writer.
def _q_to_wdl(q):
    """Approximate WDL from scalar Q in [-1, 1].

    Uses the same draw-synthesis formula as C++ game_manager.cpp:
    d = 0.5*(1-Q²), then W and L are scaled proportionally.
    This keeps Python-side played_eval/best_eval consistent with
    root_wdl from C++, which the resign calibrator depends on.
    """
    q = max(-1.0, min(1.0, q))
    w_raw = 0.5 * (1.0 + q)
    l_raw = 0.5 * (1.0 - q)
    d = 0.5 * (1.0 - q * q)
    scale = 1.0 - d
    return (scale * w_raw, d, scale * l_raw)


# Backward-compatibility alias. Existing callers (tests, play_games_batched's
# legacy batch path) continue to import `GameLoopManager`. Remove once the
# continuous-flow mode has been stable for several generations.
GameLoopManager = GamePoolManager
