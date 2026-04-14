"""Phase B: Stockfish multipv distillation data.

Walks a PGN file (or a list of FENs), runs Stockfish at fixed depth with
multipv=N on each sampled position, and produces format-v2 .npz shards with:

  - soft policy target = softmax(cp_scores / temperature) over the top-N PV moves
  - WDL value target   = (W, D, L) derived from cp via Lc0-style conversion
  - moves_left target  = plies remaining in the game (if PGN) or NaN

Output shards are compatible with `ChessDataset` and can be fed to
`training.pretrain --soft-policy-weight 0.2 --resume-from <phase_a.pt>`.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import time
from pathlib import Path

import chess
import chess.pgn
import numpy as np

from training.dataset import _extract_metadata_from_dense, _pack_dense_planes
from training.encoder import POLICY_SIZE, encode_board, mirror_move, move_to_index
from training.pretrain_dataset import (
    ShardBuffer,
    headers_pass_filters,
    iter_pgn_games,
    open_pgn_stream,
)


# ---------- Stockfish UCI driver ----------

INFO_RE = re.compile(
    r"multipv\s+(\d+).*?score\s+(cp|mate)\s+(-?\d+).*?\bpv\s+([a-h][1-8][a-h][1-8][qrbn]?)"
)


class StockfishMultiPV:
    """Thin wrapper around a Stockfish subprocess speaking UCI with MultiPV."""

    def __init__(self, binary: str, threads: int = 4, hash_mb: int = 512, multipv: int = 10):
        binary = os.path.abspath(binary)
        if not os.path.isfile(binary):
            raise FileNotFoundError(f"Stockfish binary not found: {binary}")
        self.proc = subprocess.Popen(
            [binary],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self.multipv = multipv
        self._send('uci')
        self._read_until('uciok')
        self._send(f'setoption name Threads value {threads}')
        self._send(f'setoption name Hash value {hash_mb}')
        self._send(f'setoption name MultiPV value {multipv}')
        self._send('isready')
        self._read_until('readyok')

    def _send(self, cmd: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(cmd + '\n')
        self.proc.stdin.flush()

    def _read_until(self, token: str) -> list[str]:
        assert self.proc.stdout is not None
        lines: list[str] = []
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("stockfish closed stdout unexpectedly")
            lines.append(line.rstrip('\n'))
            if token in line:
                return lines

    def eval_position(self, fen: str, depth: int) -> list[tuple[str, int]]:
        """Return up to `multipv` (uci_move, cp_score) pairs from Stockfish.

        cp_score is from side-to-move perspective. Mate scores are mapped to
        large cp values: +30000 - ply for a mate we give, -30000 + ply for a
        mate we receive (Lc0 convention).
        """
        self._send(f'position fen {fen}')
        self._send(f'go depth {depth}')

        # Collect the last "info" line per multipv index until we see bestmove.
        latest: dict[int, tuple[int, str]] = {}  # multipv -> (cp, uci)
        assert self.proc.stdout is not None
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("stockfish closed stdout unexpectedly")
            if line.startswith('bestmove'):
                break
            if not line.startswith('info'):
                continue
            m = INFO_RE.search(line)
            if m is None:
                continue
            mpv = int(m.group(1))
            score_kind = m.group(2)
            val = int(m.group(3))
            uci = m.group(4)
            if score_kind == 'mate':
                cp = (30000 - abs(val)) * (1 if val > 0 else -1)
            else:
                cp = val
            latest[mpv] = (cp, uci)

        # Sort by multipv rank (1 = best).
        out = []
        for mpv in sorted(latest.keys()):
            cp, uci = latest[mpv]
            out.append((uci, cp))
        return out

    def close(self) -> None:
        try:
            self._send('quit')
        except Exception:
            pass
        try:
            self.proc.wait(timeout=2)
        except Exception:
            self.proc.kill()


# ---------- cp → policy / WDL conversions ----------

def cp_to_wdl(cp: int, draw_scale: float = 200.0) -> tuple[float, float, float]:
    """Lc0-style cp → WDL. P(win) = sigmoid(cp/scale), draw mass declines
    sharply as |cp| grows. Simple 3-term model: P(W) + P(D) + P(L) = 1.
    """
    # Sigmoid expected score in [0, 1].
    expected = 1.0 / (1.0 + math.exp(-cp / draw_scale))
    # Draw probability peaked at cp=0; falls off with |cp|.
    # Chosen to roughly match empirical engine draw rates at 3200+ Elo.
    draw = max(0.0, 1.0 - abs(cp) / 300.0) * 0.5
    win = max(0.0, expected - draw / 2.0)
    loss = max(0.0, 1.0 - expected - draw / 2.0)
    s = win + draw + loss
    if s <= 0:
        return (0.0, 1.0, 0.0)
    return (win / s, draw / s, loss / s)


def cp_scores_to_policy(
    pv: list[tuple[str, int]], board: chess.Board, temperature_cp: float = 80.0,
) -> np.ndarray:
    """Convert (move, cp) pairs to a 1858-dim soft policy target.

    Higher cp_score ⇒ larger probability. Temperature controls sharpness
    (lower = sharper). Moves outside the 1858 encoding are dropped.
    """
    policy = np.zeros(POLICY_SIZE, dtype=np.float32)
    if not pv:
        return policy

    indices: list[int] = []
    scores: list[float] = []
    for uci, cp in pv:
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            continue
        idx = _move_to_policy_index(move, board)
        if idx is None:
            continue
        indices.append(idx)
        scores.append(cp / temperature_cp)

    if not indices:
        return policy

    s = np.asarray(scores, dtype=np.float64)
    s = s - s.max()
    w = np.exp(s)
    w = w / w.sum()
    for idx, p in zip(indices, w):
        policy[idx] = float(p)
    return policy


def _move_to_policy_index(move: chess.Move, board: chess.Board) -> int | None:
    is_white = (board.turn == chess.WHITE)
    from_sq = move.from_square if is_white else mirror_move(move.from_square)
    to_sq = move.to_square if is_white else mirror_move(move.to_square)
    promo = None
    if move.promotion is not None:
        if move.promotion == chess.QUEEN:
            promo = None
        elif move.promotion == chess.KNIGHT:
            promo = 'n'
        elif move.promotion == chess.BISHOP:
            promo = 'b'
        elif move.promotion == chess.ROOK:
            promo = 'r'
    return move_to_index(from_sq, to_sq, promo)


# ---------- Shard writer (reuses phase A buffer but with soft policy) ----------

def add_soft_sample(
    buf: ShardBuffer,
    board: chess.Board,
    policy: np.ndarray,
    wdl: tuple[float, float, float],
    plies_remaining: float,
) -> None:
    """Emit one position with a pre-built soft policy vector into `buf`."""
    dense = encode_board(board)
    dense_n = dense[None, ...]
    bb = _pack_dense_planes(dense_n)[0]
    meta = _extract_metadata_from_dense(dense_n)

    buf.bitboards.append(bb)
    buf.stm.append(bool(meta['stm'][0]))
    buf.castling.append(int(meta['castling'][0]))
    buf.rule50.append(int(meta['rule50'][0]))
    buf.fullmove.append(int(meta['fullmove'][0]))
    buf.policies.append(policy)
    buf.values.append(wdl)
    buf.moves_left.append(float(plies_remaining))


# ---------- PGN walker that queries Stockfish per sampled ply ----------

def label_pgn(
    pgn_path: Path,
    out_dir: Path,
    stockfish_binary: str,
    min_elo: int = 2400,
    min_base_s: int = 480,
    multipv: int = 10,
    depth: int = 15,
    threads: int = 4,
    hash_mb: int = 512,
    temperature_cp: float = 80.0,
    shard_size: int = 50_000,
    positions_per_game: int | None = 3,
    max_positions: int | None = None,
    max_games: int | None = None,
    skip_opening_plies: int = 10,
    start_shard: int = 0,
) -> None:
    """Walk `pgn_path`; for each qualifying game, sample positions and label
    them with Stockfish multipv. Writes shards under `out_dir`.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    buf = ShardBuffer(out_dir=out_dir, shard_idx=start_shard, max_positions=shard_size)
    sf = StockfishMultiPV(stockfish_binary, threads=threads, hash_mb=hash_mb, multipv=multipv)

    allowed_results = {'1-0', '0-1', '1/2-1/2'}
    games_read = 0
    games_kept = 0
    positions_total = 0
    t0 = time.time()

    rng = np.random.default_rng(0xC0FFEE)

    try:
        with open_pgn_stream(pgn_path) as stream:
            for game in iter_pgn_games(stream):
                games_read += 1
                if not headers_pass_filters(game.headers, min_elo, min_base_s, allowed_results):
                    continue

                board = game.board()
                mainline = list(game.mainline_moves())
                total_plies = len(mainline)
                if total_plies <= skip_opening_plies + 1:
                    continue

                # Choose which plies to label in this game.
                eligible = list(range(skip_opening_plies, total_plies))
                if positions_per_game is not None and len(eligible) > positions_per_game:
                    chosen = set(rng.choice(eligible, size=positions_per_game, replace=False).tolist())
                else:
                    chosen = set(eligible)

                kept_this_game = False
                for ply_idx, move in enumerate(mainline):
                    if ply_idx in chosen:
                        fen = board.fen()
                        pv = sf.eval_position(fen, depth=depth)
                        if pv:
                            policy = cp_scores_to_policy(pv, board, temperature_cp=temperature_cp)
                            if policy.sum() > 0:
                                # best-move cp → WDL target
                                top_cp = pv[0][1]
                                wdl = cp_to_wdl(top_cp)
                                plies_remaining = total_plies - ply_idx - 1
                                add_soft_sample(buf, board, policy, wdl, plies_remaining)
                                positions_total += 1
                                kept_this_game = True

                                if len(buf) >= shard_size:
                                    written = buf.flush()
                                    if written:
                                        print(f"  wrote {written.name} (total {positions_total} pos)")

                    if move not in board.legal_moves:
                        break
                    board.push(move)

                if kept_this_game:
                    games_kept += 1

                if (games_kept % 50) == 0 and games_kept > 0:
                    rate = positions_total / max(time.time() - t0, 1e-6)
                    print(
                        f"  {games_read:,} scanned / {games_kept:,} kept / "
                        f"{positions_total:,} pos ({rate:.1f} pos/s)"
                    )

                if max_games is not None and games_read >= max_games:
                    break
                if max_positions is not None and positions_total >= max_positions:
                    break
    finally:
        written = buf.flush()
        if written:
            print(f"  wrote {written.name} (final)")
        sf.close()

    elapsed = time.time() - t0
    print(
        f"\nDone: {games_read:,} games scanned, {games_kept:,} kept, "
        f"{positions_total:,} positions labeled in {elapsed:.1f}s"
    )


def main():
    parser = argparse.ArgumentParser(description='Stockfish multipv distillation labeler (phase B)')
    parser.add_argument('--pgn', type=Path, required=True, help='PGN or .pgn.zst source')
    parser.add_argument('--out-dir', type=Path, default=Path('pretrain_data/phase_b'))
    parser.add_argument('--stockfish', default='engines/stockfish/stockfish-windows-x86-64-avx2.exe',
                        help='Stockfish binary path')
    parser.add_argument('--min-elo', type=int, default=2400)
    parser.add_argument('--min-base-s', type=int, default=480)
    parser.add_argument('--multipv', type=int, default=10)
    parser.add_argument('--depth', type=int, default=15)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--hash-mb', type=int, default=512)
    parser.add_argument('--temperature-cp', type=float, default=80.0,
                        help='Softmax temperature in centipawns (lower = sharper policy)')
    parser.add_argument('--shard-size', type=int, default=50_000)
    parser.add_argument('--positions-per-game', type=int, default=3,
                        help='Sampled positions per game (None = every ply)')
    parser.add_argument('--max-positions', type=int, default=None)
    parser.add_argument('--max-games', type=int, default=None)
    parser.add_argument('--skip-opening-plies', type=int, default=10)
    parser.add_argument('--start-shard', type=int, default=0)
    args = parser.parse_args()

    label_pgn(
        pgn_path=args.pgn,
        out_dir=args.out_dir,
        stockfish_binary=args.stockfish,
        min_elo=args.min_elo,
        min_base_s=args.min_base_s,
        multipv=args.multipv,
        depth=args.depth,
        threads=args.threads,
        hash_mb=args.hash_mb,
        temperature_cp=args.temperature_cp,
        shard_size=args.shard_size,
        positions_per_game=args.positions_per_game,
        max_positions=args.max_positions,
        max_games=args.max_games,
        skip_opening_plies=args.skip_opening_plies,
        start_shard=args.start_shard,
    )


if __name__ == '__main__':
    main()
