"""Build supervised-pretraining shards from a Lichess PGN dump.

Phase A: one-hot policy = move actually played, value = game result (WDL
from side-to-move perspective), moves_left = plies remaining in the game.

Reads a `.pgn.zst` (or `.pgn`) stream, filters games by Elo and time control,
then encodes every position as a 112x8x8 tensor via `encode_board` (so the
network sees real 8-step history, matching self-play).

Output shards are format-v2 .npz files compatible with `ChessDataset` — same
layout selfplay.py writes, so `pretrain.py` can train on them with no code
changes elsewhere.
"""

from __future__ import annotations

import argparse
import io
import os
import time
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.pgn
import numpy as np

from training.dataset import _extract_metadata_from_dense, _pack_dense_planes
from training.encoder import POLICY_SIZE, encode_board, mirror_move, move_to_index


def move_to_policy_index(move: chess.Move, board: chess.Board) -> int | None:
    """Map a python-chess Move (on `board`) to a 1858-dim policy index.

    The encoder is always oriented from side-to-move perspective, so we mirror
    squares vertically when Black is to move. Queen promotions use None;
    N/B/R promotions use their letter.
    """
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


def result_to_wdl(result: str, stm_is_white: bool) -> tuple[float, float, float]:
    """Convert PGN result string to WDL tuple from side-to-move perspective."""
    if result == '1-0':
        return (1.0, 0.0, 0.0) if stm_is_white else (0.0, 0.0, 1.0)
    if result == '0-1':
        return (0.0, 0.0, 1.0) if stm_is_white else (1.0, 0.0, 0.0)
    return (0.0, 1.0, 0.0)


# Lichess TimeControl headers are "base+inc" in seconds, e.g. "600+5".
# Classical: base >= 1500s (25+). Rapid: 480-1499s (8-24 min). Blitz: 180-479s.
# Bullet: < 180s. We want only rapid + classical for the strong-player supply.
def time_control_ok(tc: str, min_base_s: int = 480) -> bool:
    if not tc or tc in ('-', '?'):
        return False
    try:
        base = int(tc.split('+', 1)[0])
    except ValueError:
        return False
    return base >= min_base_s


@dataclass
class ShardBuffer:
    """In-memory accumulator for one output shard. Grows per-position; flushes
    when `max_positions` is reached."""
    out_dir: Path
    shard_idx: int = 0
    max_positions: int = 100_000

    def __post_init__(self):
        self._reset()

    def _reset(self):
        self.bitboards: list[np.ndarray] = []   # each (104,) uint64
        self.stm: list[bool] = []
        self.castling: list[int] = []
        self.rule50: list[int] = []
        self.fullmove: list[int] = []
        self.policies: list[np.ndarray] = []    # each (1858,) float32
        self.values: list[tuple[float, float, float]] = []
        self.moves_left: list[float] = []

    def add(
        self,
        board: chess.Board,
        policy_idx: int,
        wdl: tuple[float, float, float],
        plies_remaining: int,
    ) -> None:
        # Dense encode then pack in-place to avoid keeping a (112,8,8) float32
        # tensor per position in the shard buffer.
        dense = encode_board(board)                 # (112, 8, 8) float32
        dense_n = dense[None, ...]                  # (1, 112, 8, 8) for existing helpers
        bb = _pack_dense_planes(dense_n)[0]         # (104,) uint64
        meta = _extract_metadata_from_dense(dense_n)

        policy = np.zeros(POLICY_SIZE, dtype=np.float32)
        policy[policy_idx] = 1.0

        self.bitboards.append(bb)
        self.stm.append(bool(meta['stm'][0]))
        self.castling.append(int(meta['castling'][0]))
        self.rule50.append(int(meta['rule50'][0]))
        self.fullmove.append(int(meta['fullmove'][0]))
        self.policies.append(policy)
        self.values.append(wdl)
        self.moves_left.append(float(plies_remaining))

    def __len__(self) -> int:
        return len(self.bitboards)

    def flush(self) -> Path | None:
        if not self.bitboards:
            return None
        n = len(self.bitboards)
        out_path = self.out_dir / f"shard_{self.shard_idx:05d}.npz"

        np.savez_compressed(
            out_path,
            format_version=np.uint8(2),
            bitboards=np.stack(self.bitboards, axis=0).astype(np.uint64),
            stm=np.asarray(self.stm, dtype=np.bool_),
            castling=np.asarray(self.castling, dtype=np.uint8),
            rule50=np.asarray(self.rule50, dtype=np.uint8),
            fullmove=np.asarray(self.fullmove, dtype=np.uint16),
            policies=np.stack(self.policies, axis=0).astype(np.float32),
            values=np.asarray(self.values, dtype=np.float32),
            moves_left=np.asarray(self.moves_left, dtype=np.float32),
            use_policy=np.ones(n, dtype=np.bool_),
        )
        self.shard_idx += 1
        self._reset()
        return out_path


def iter_pgn_games(stream: io.TextIOBase):
    """Yield `chess.pgn.Game` objects from a text-mode stream until EOF."""
    while True:
        game = chess.pgn.read_game(stream)
        if game is None:
            return
        yield game


def open_pgn_stream(path: Path) -> io.TextIOBase:
    """Open `.pgn` or `.pgn.zst` as a line-buffered UTF-8 text stream."""
    if path.suffix == '.zst':
        import zstandard
        dctx = zstandard.ZstdDecompressor(max_window_size=2**31)
        raw = path.open('rb')
        reader = dctx.stream_reader(raw)
        return io.TextIOWrapper(reader, encoding='utf-8', errors='replace')
    return path.open('r', encoding='utf-8', errors='replace')


def headers_pass_filters(
    headers: chess.pgn.Headers,
    min_elo: int,
    min_base_s: int,
    allowed_results: set[str],
) -> bool:
    result = headers.get('Result', '*')
    if result not in allowed_results:
        return False
    try:
        we = int(headers.get('WhiteElo', '0'))
        be = int(headers.get('BlackElo', '0'))
    except ValueError:
        return False
    if we < min_elo or be < min_elo:
        return False
    if not time_control_ok(headers.get('TimeControl', ''), min_base_s=min_base_s):
        return False
    if headers.get('Termination', '') == 'Abandoned':
        return False
    return True


def process_game(game: chess.pgn.Game, buf: ShardBuffer) -> int:
    """Walk a game's mainline, emitting one training position per ply.

    Returns the number of positions added.
    """
    result = game.headers.get('Result', '*')
    board = game.board()
    mainline = list(game.mainline_moves())
    total_plies = len(mainline)
    if total_plies == 0:
        return 0

    added = 0
    for ply_idx, move in enumerate(mainline):
        # Skip null / illegal moves defensively; the stream may be malformed.
        if move is None or move not in board.legal_moves:
            break
        policy_idx = move_to_policy_index(move, board)
        if policy_idx is None:
            # Unrepresentable move (shouldn't happen with legal chess moves)
            board.push(move)
            continue

        stm_is_white = (board.turn == chess.WHITE)
        wdl = result_to_wdl(result, stm_is_white)
        plies_remaining = total_plies - ply_idx - 1
        buf.add(board, policy_idx, wdl, plies_remaining)
        added += 1

        board.push(move)

    return added


def build_shards(
    pgn_path: Path,
    out_dir: Path,
    min_elo: int = 2400,
    min_base_s: int = 480,
    max_positions: int | None = None,
    shard_size: int = 100_000,
    max_games: int | None = None,
    start_shard: int = 0,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    buf = ShardBuffer(out_dir=out_dir, shard_idx=start_shard, max_positions=shard_size)
    allowed_results = {'1-0', '0-1', '1/2-1/2'}

    games_read = 0
    games_kept = 0
    positions_total = 0
    t0 = time.time()
    last_report = t0

    with open_pgn_stream(pgn_path) as stream:
        for game in iter_pgn_games(stream):
            games_read += 1

            if headers_pass_filters(game.headers, min_elo, min_base_s, allowed_results):
                added = process_game(game, buf)
                if added > 0:
                    games_kept += 1
                    positions_total += added

                    while len(buf) >= shard_size:
                        # Flush only up to shard_size at a time — but our buffer
                        # grows per-game, so once it crosses the threshold we
                        # flush the whole buffer (slight overshoot is fine).
                        written = buf.flush()
                        if written:
                            print(f"  wrote {written.name} ({positions_total} pos, {games_kept} games kept)")

            if (games_read % 5000) == 0:
                now = time.time()
                if now - last_report > 5:
                    rate = games_read / (now - t0)
                    print(
                        f"  scanned {games_read:,} games "
                        f"({games_kept:,} kept, {positions_total:,} pos, {rate:.0f} games/s)"
                    )
                    last_report = now

            if max_games is not None and games_read >= max_games:
                break
            if max_positions is not None and positions_total >= max_positions:
                break

    # Final flush for the partial tail shard.
    written = buf.flush()
    if written:
        print(f"  wrote {written.name} (final)")

    elapsed = time.time() - t0
    print(
        f"\nDone: scanned {games_read:,} games, kept {games_kept:,}, "
        f"produced {positions_total:,} positions in {elapsed:.1f}s"
    )


def main():
    parser = argparse.ArgumentParser(description="Build supervised-pretraining shards from Lichess PGN.")
    parser.add_argument('--pgn', type=Path, required=True,
                        help='Path to .pgn or .pgn.zst file')
    parser.add_argument('--out-dir', type=Path, default=Path('pretrain_data/phase_a'),
                        help='Shard output directory')
    parser.add_argument('--min-elo', type=int, default=2400,
                        help='Minimum Elo for both players (default 2400)')
    parser.add_argument('--min-base-s', type=int, default=480,
                        help='Minimum time control base in seconds (default 480 = rapid+classical)')
    parser.add_argument('--shard-size', type=int, default=100_000,
                        help='Positions per output shard')
    parser.add_argument('--max-positions', type=int, default=None,
                        help='Stop after this many positions total (for smoke tests)')
    parser.add_argument('--max-games', type=int, default=None,
                        help='Stop after scanning this many games total (for smoke tests)')
    parser.add_argument('--start-shard', type=int, default=0,
                        help='First shard index (use when appending to existing output)')
    args = parser.parse_args()

    build_shards(
        pgn_path=args.pgn,
        out_dir=args.out_dir,
        min_elo=args.min_elo,
        min_base_s=args.min_base_s,
        max_positions=args.max_positions,
        shard_size=args.shard_size,
        max_games=args.max_games,
        start_shard=args.start_shard,
    )


if __name__ == '__main__':
    main()
