"""Build supervised-pretraining shards from the chess.com GM dataset.

Reads a local CSV (or a .zip containing one), deduplicates games by URL
(each game appears twice when both players are GMs), filters on rating +
termination + time control, then encodes positions in parallel worker
processes. Output shards are format-v2 .npz files consumable by
`pretrain.py` with no other changes.

Performance design:
  - CSV is streamed in `chunk_size` row chunks (pandas `chunksize=`).
  - Only the 7 columns we actually use are loaded (`usecols=[...]`).
  - URL dedup is a set (O(1) lookup, O(#unique-urls) memory — ~400MB for 4M).
  - Main process does all filtering (fast); workers only do the expensive
    step: PGN parse + per-ply `encode_board` (~0.5-1ms/pos).
  - Workers return per-game arrays with `policy_idx` as int32 (not dense
    1858-float) — cuts inter-process transfer by ~200×.
  - Shards flush via one `np.concatenate` per field; dense policy tensor
    materialized once at flush time from the int32 indices.

Completeness design:
  - `IngestStats` tracks every drop reason as a separate counter (9
    categories). Final accounting asserts dropped + kept == scanned and
    warns on mismatch.
  - Rows with missing URL still get processed (dedup is optional per row).
  - Chess960 / variant games are dropped (the encoder assumes standard start).
"""
from __future__ import annotations

import argparse
import contextlib
import io
import math
import multiprocessing as mp
import os
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import chess
import chess.pgn
import chess.polyglot
import numpy as np
import pandas as pd

from training.dataset import _extract_metadata_from_dense, _pack_dense_planes
from training.encoder import POLICY_SIZE, encode_board
from training.pretrain_dataset import (
    move_to_policy_index,
    result_to_wdl,
    time_control_ok,
)


# ---------- Worker (runs in child process) ----------

# Return-status codes. Tuple-tagging the return value lets the main process
# distinguish parse failures from empty mainlines without overloading None.
_WORKER_OK = 0
_WORKER_PARSE_FAILED = 1
_WORKER_EMPTY = 2

# Populated by `_init_worker` at Pool startup. Read-only after init.
_WORKER_CFG: dict = {
    'skip_opening': 0,
    'skip_end': 0,
    'positions_per_game': 0,  # 0 = no limit
}


def _init_worker(skip_opening: int, skip_end: int, positions_per_game: int,
                 seed_base: int) -> None:
    """Pool initializer: cache sampling config + seed RNG per worker process."""
    _WORKER_CFG['skip_opening'] = int(skip_opening)
    _WORKER_CFG['skip_end'] = int(skip_end)
    _WORKER_CFG['positions_per_game'] = int(positions_per_game)
    # Per-process seed so workers don't all draw identical samples.
    np.random.seed((seed_base + os.getpid()) & 0xFFFFFFFF)


def _encode_game_worker(task: tuple[str, str]):
    """Parse PGN + walk mainline + encode every ply. Returns (status, payload).

    Payload on _WORKER_OK is a per-game dict of numpy arrays with policy
    stored as int32 indices (not dense). The main process materializes the
    dense (N, 1858) policy tensor at flush time.
    """
    pgn_text, result = task
    try:
        game = chess.pgn.read_game(io.StringIO(pgn_text))
    except Exception:
        return (_WORKER_PARSE_FAILED, None)
    if game is None:
        return (_WORKER_PARSE_FAILED, None)

    mainline = list(game.mainline_moves())
    total_plies = len(mainline)
    if total_plies == 0:
        return (_WORKER_EMPTY, None)

    # Sampling: eligible window is [skip_opening, total_plies - skip_end).
    # Maia skips 10 opening plies, Stockfish NNUE skips 16 (write_minply);
    # we default to 12. End-skip cuts resignation/time-trouble tails.
    # AlphaZero caps at 30 positions per game to decorrelate samples.
    skip_o = _WORKER_CFG['skip_opening']
    skip_e = _WORKER_CFG['skip_end']
    k = _WORKER_CFG['positions_per_game']

    lo = min(skip_o, total_plies)
    hi = max(lo, total_plies - skip_e)
    if lo >= hi:
        return (_WORKER_EMPTY, None)

    if k > 0 and (hi - lo) > k:
        chosen_arr = np.random.choice(np.arange(lo, hi), size=k, replace=False)
        chosen = set(int(x) for x in chosen_arr)
    else:
        chosen = set(range(lo, hi))

    board = game.board()
    bb_list, stm_list, cast_list, r50_list, fm_list = [], [], [], [], []
    pidx_list, val_list, ml_list, zob_list = [], [], [], []

    for ply_idx, move in enumerate(mainline):
        if move is None or move not in board.legal_moves:
            break

        if ply_idx in chosen:
            policy_idx = move_to_policy_index(move, board)
            if policy_idx is not None:
                stm_is_white = (board.turn == chess.WHITE)
                wdl = result_to_wdl(result, stm_is_white)
                plies_remaining = total_plies - ply_idx - 1

                dense = encode_board(board)                 # (112, 8, 8) float32
                dense_n = dense[None, ...]                  # (1, 112, 8, 8)
                bb = _pack_dense_planes(dense_n)[0]         # (104,) uint64
                meta = _extract_metadata_from_dense(dense_n)

                bb_list.append(bb)
                stm_list.append(bool(meta['stm'][0]))
                cast_list.append(int(meta['castling'][0]))
                r50_list.append(int(meta['rule50'][0]))
                fm_list.append(int(meta['fullmove'][0]))
                pidx_list.append(int(policy_idx))
                val_list.append(wdl)
                ml_list.append(float(plies_remaining))
                zob_list.append(chess.polyglot.zobrist_hash(board))

        board.push(move)

    if not bb_list:
        return (_WORKER_EMPTY, None)

    return (_WORKER_OK, {
        'bitboards': np.stack(bb_list, axis=0).astype(np.uint64),
        'stm': np.asarray(stm_list, dtype=np.bool_),
        'castling': np.asarray(cast_list, dtype=np.uint8),
        'rule50': np.asarray(r50_list, dtype=np.uint8),
        'fullmove': np.asarray(fm_list, dtype=np.uint16),
        'policy_idx': np.asarray(pidx_list, dtype=np.int32),
        'values': np.asarray(val_list, dtype=np.float32),
        'moves_left': np.asarray(ml_list, dtype=np.float32),
        'zobrist': np.asarray(zob_list, dtype=np.uint64),
    })


# ---------- Shard buffer ----------

@dataclass
class BatchedShardBuffer:
    """Accumulates per-game numpy batches and flushes a single shard.

    Faster than the per-position `ShardBuffer` in `pretrain_dataset.py`
    because (a) workers produce numpy batches directly, (b) flush is one
    `np.concatenate` per field instead of a stack over 100k per-position
    lists, and (c) dense (N, 1858) policies are materialized only once at
    flush time from the compact int32 index arrays.
    """
    out_dir: Path
    shard_idx: int = 0

    def __post_init__(self):
        self._batches: list[dict] = []
        self._count: int = 0

    def add_game(self, arrays: dict) -> None:
        self._batches.append(arrays)
        self._count += int(arrays['bitboards'].shape[0])

    def __len__(self) -> int:
        return self._count

    def flush(self) -> Path | None:
        if not self._batches:
            return None

        total = self._count
        # Dense policies built here rather than per-position in workers.
        policies = np.zeros((total, POLICY_SIZE), dtype=np.float32)
        offset = 0
        for b in self._batches:
            n = int(b['bitboards'].shape[0])
            policies[np.arange(offset, offset + n), b['policy_idx']] = 1.0
            offset += n

        out_path = self.out_dir / f"shard_{self.shard_idx:05d}.npz"
        np.savez_compressed(
            out_path,
            format_version=np.uint8(2),
            bitboards=np.concatenate([b['bitboards'] for b in self._batches], axis=0),
            stm=np.concatenate([b['stm'] for b in self._batches], axis=0),
            castling=np.concatenate([b['castling'] for b in self._batches], axis=0),
            rule50=np.concatenate([b['rule50'] for b in self._batches], axis=0),
            fullmove=np.concatenate([b['fullmove'] for b in self._batches], axis=0),
            policies=policies,
            values=np.concatenate([b['values'] for b in self._batches], axis=0),
            moves_left=np.concatenate([b['moves_left'] for b in self._batches], axis=0),
            use_policy=np.ones(total, dtype=np.bool_),
        )
        self.shard_idx += 1
        self._batches = []
        self._count = 0
        return out_path


# ---------- Stats ----------

@dataclass
class IngestStats:
    rows_scanned: int = 0
    duplicate_url: int = 0
    missing_pgn: int = 0
    invalid_result: int = 0
    abandoned: int = 0
    below_rating: int = 0
    failed_tc: int = 0
    variant_or_chess960: int = 0
    pgn_parse_failed: int = 0
    empty_mainline: int = 0
    games_kept: int = 0
    positions_kept: int = 0
    duplicate_position: int = 0  # corpus-wide zobrist collisions

    def _dropped_total(self) -> int:
        return (
            self.duplicate_url + self.missing_pgn + self.invalid_result
            + self.abandoned + self.below_rating + self.failed_tc
            + self.variant_or_chess960 + self.pgn_parse_failed
            + self.empty_mainline
        )

    def report(self, elapsed: float, out=sys.stdout) -> None:
        fields = [
            ("rows_scanned",            self.rows_scanned),
            ("duplicate_url (dropped)", self.duplicate_url),
            ("missing_pgn (dropped)",   self.missing_pgn),
            ("invalid_result (dropped)", self.invalid_result),
            ("abandoned (dropped)",     self.abandoned),
            ("below_rating (dropped)",  self.below_rating),
            ("failed_tc (dropped)",     self.failed_tc),
            ("variant/chess960 (dropped)", self.variant_or_chess960),
            ("pgn_parse_failed (dropped)", self.pgn_parse_failed),
            ("empty_mainline (dropped)", self.empty_mainline),
            ("duplicate_position (dropped)", self.duplicate_position),
            ("GAMES KEPT",              self.games_kept),
            ("POSITIONS KEPT",          self.positions_kept),
        ]
        for k, v in fields:
            print(f"  {k:32s}: {v:,}", file=out)
        print(f"  elapsed: {elapsed:.1f}s", file=out)
        accounted = self._dropped_total() + self.games_kept
        if accounted != self.rows_scanned:
            diff = self.rows_scanned - accounted
            print(
                f"  WARN: rows_scanned={self.rows_scanned:,} but "
                f"dropped+kept={accounted:,} (diff={diff:+,}). "
                "Some rows fell through the filter tree unclassified.",
                file=sys.stderr,
            )


# ---------- Main-process filtering (streaming) ----------

def _parse_int(v: object) -> int | None:
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    try:
        return int(v)
    except (ValueError, TypeError):
        return None


def _tc_ok(tc: str, min_base_s: int, allow_daily: bool) -> bool:
    """Extend `time_control_ok` with chess.com's daily format ('1/<sec-per-move>').

    Daily games are correspondence — effectively unlimited thinking time.
    pretrain_dataset.time_control_ok would reject '1/86400' because
    `int(tc.split('+',1)[0])` returns 1.
    """
    if not tc or tc in ('-', '?'):
        return False
    tc = tc.strip()
    if tc.startswith('1/'):
        return allow_daily
    return time_control_ok(tc, min_base_s=min_base_s)


def _is_chess960(pgn_text: str) -> bool:
    head = pgn_text[:512]
    return 'Chess960' in head or 'Fischerandom' in head


_CSV_USECOLS = ['url', 'pgn', 'Result', 'Termination',
                'WhiteElo', 'BlackElo', 'TimeControl']


def _pick_inner_csv(zf: zipfile.ZipFile) -> zipfile.ZipInfo:
    csvs = [i for i in zf.infolist() if i.filename.lower().endswith('.csv')]
    if not csvs:
        raise FileNotFoundError("No .csv inside archive")
    csvs.sort(key=lambda i: i.file_size, reverse=True)  # main games table = largest
    return csvs[0]


@contextlib.contextmanager
def _open_csv_reader(csv_path: Path, chunk_size: int):
    """Yield a chunked pandas reader over a .csv or .zip-containing-csv.

    Owns the zip-file handle lifetime so it stays open while the reader is
    iterated. Pandas chunks read lazily from the underlying text stream —
    works for the 15GB chess.com CSV without materializing it in memory.
    """
    if csv_path.suffix.lower() == '.zip':
        with zipfile.ZipFile(csv_path) as zf:
            inner = _pick_inner_csv(zf)
            print(f"Streaming {inner.filename} ({inner.file_size:,} bytes) "
                  f"from {csv_path.name}")
            with zf.open(inner) as raw, \
                 io.TextIOWrapper(raw, encoding='utf-8', errors='replace') as text:
                yield pd.read_csv(
                    text, usecols=_CSV_USECOLS,
                    chunksize=chunk_size, low_memory=False,
                )
    else:
        yield pd.read_csv(
            csv_path, usecols=_CSV_USECOLS,
            chunksize=chunk_size, low_memory=False,
        )


def _iter_csv_tasks(
    reader,
    stats: IngestStats,
    seen_urls: set[str],
    *,
    min_elo: int,
    min_base_s: int,
    allow_daily: bool,
) -> Iterator[tuple[str, str]]:
    """Yield (pgn_text, result) for rows that pass filters.

    All filter drops bump the corresponding `stats` counter here; workers
    further classify parse_failed / empty_mainline.
    """
    allowed_results = {'1-0', '0-1', '1/2-1/2'}
    for chunk in reader:
        for row in chunk.itertuples(index=False):
            stats.rows_scanned += 1

            url = getattr(row, 'url', None)
            if isinstance(url, str) and url:
                if url in seen_urls:
                    stats.duplicate_url += 1
                    continue
                seen_urls.add(url)
            # Missing url → process the row (no dedup guarantee possible).

            pgn_text = getattr(row, 'pgn', None)
            if not isinstance(pgn_text, str) or not pgn_text.strip():
                stats.missing_pgn += 1
                continue

            result = getattr(row, 'Result', None)
            if result not in allowed_results:
                stats.invalid_result += 1
                continue

            termination = getattr(row, 'Termination', '')
            if isinstance(termination, str) and 'abandoned' in termination.lower():
                stats.abandoned += 1
                continue

            if min_elo > 0:
                we = _parse_int(getattr(row, 'WhiteElo', None))
                be = _parse_int(getattr(row, 'BlackElo', None))
                if we is None or be is None or we < min_elo or be < min_elo:
                    stats.below_rating += 1
                    continue

            tc = getattr(row, 'TimeControl', '')
            if not isinstance(tc, str):
                tc = str(tc) if tc is not None else ''
            if not _tc_ok(tc, min_base_s=min_base_s, allow_daily=allow_daily):
                stats.failed_tc += 1
                continue

            if _is_chess960(pgn_text):
                stats.variant_or_chess960 += 1
                continue

            yield (pgn_text, result)


# ---------- Driver ----------

def _apply_worker_result(
    status: int,
    payload,
    buf: BatchedShardBuffer,
    stats: IngestStats,
    seen_zobrist: set[int] | None,
) -> None:
    if status == _WORKER_OK:
        stats.games_kept += 1
        zob = payload.pop('zobrist')  # strip so it never hits the shard file
        if seen_zobrist is not None:
            # Corpus-wide dedup: drop positions whose zobrist we've already kept.
            mask = np.fromiter(
                (int(z) not in seen_zobrist for z in zob),
                dtype=np.bool_, count=len(zob),
            )
            kept = int(mask.sum())
            dups = len(zob) - kept
            if dups:
                stats.duplicate_position += dups
                if kept == 0:
                    return
                for key in list(payload.keys()):
                    payload[key] = payload[key][mask]
            # Add the surviving hashes to the corpus-wide set.
            seen_zobrist.update(int(z) for z in zob[mask])
        stats.positions_kept += int(payload['bitboards'].shape[0])
        buf.add_game(payload)
    elif status == _WORKER_PARSE_FAILED:
        stats.pgn_parse_failed += 1
    elif status == _WORKER_EMPTY:
        stats.empty_mainline += 1


def _print_progress(stats: IngestStats, t0: float, now: float) -> None:
    elapsed = max(now - t0, 1e-6)
    print(
        f"  scanned {stats.rows_scanned:,} rows "
        f"({stats.games_kept:,} kept, {stats.positions_kept:,} pos, "
        f"{stats.rows_scanned/elapsed:.0f} rows/s, "
        f"{stats.positions_kept/elapsed:.0f} pos/s) "
        f"drops: dup_url={stats.duplicate_url:,} "
        f"dup_pos={stats.duplicate_position:,} "
        f"tc={stats.failed_tc:,} "
        f"result={stats.invalid_result:,} "
        f"abandon={stats.abandoned:,} "
        f"no_pgn={stats.missing_pgn:,} "
        f"960={stats.variant_or_chess960:,} "
        f"parse={stats.pgn_parse_failed:,} "
        f"empty={stats.empty_mainline:,}"
    )


def _consume_results(
    result_stream,
    buf: BatchedShardBuffer,
    stats: IngestStats,
    *,
    shard_size: int,
    max_positions: int | None,
    t0: float,
    seen_zobrist: set[int] | None,
) -> bool:
    """Consume (status, payload) results; flush shards; return True if early-stopped."""
    last_report = t0
    for status, payload in result_stream:
        _apply_worker_result(status, payload, buf, stats, seen_zobrist)

        while len(buf) >= shard_size:
            written = buf.flush()
            if written:
                print(
                    f"  wrote {written.name} "
                    f"({stats.positions_kept:,} pos, {stats.games_kept:,} games kept)"
                )

        if max_positions is not None and stats.positions_kept >= max_positions:
            print(f"  reached max_positions={max_positions:,}, stopping early")
            return True

        now = time.time()
        if now - last_report > 5:
            _print_progress(stats, t0, now)
            last_report = now
    return False


def build_shards_from_chesscom(
    csv_path: Path,
    out_dir: Path,
    *,
    min_elo: int,
    min_base_s: int,
    allow_daily: bool,
    shard_size: int,
    chunk_size: int,
    max_positions: int | None,
    start_shard: int,
    num_workers: int,
    pool_chunksize: int,
    skip_opening_plies: int,
    skip_end_plies: int,
    positions_per_game: int,
    dedup_positions: bool,
    seed: int,
) -> IngestStats:
    out_dir.mkdir(parents=True, exist_ok=True)
    buf = BatchedShardBuffer(out_dir=out_dir, shard_idx=start_shard)
    stats = IngestStats()
    seen_urls: set[str] = set()
    # Corpus-wide position dedup: python-chess zobrist (64-bit) → Python set.
    # ~100 bytes/entry, so ~1GB per 10M unique positions. For the full 4M-game
    # run at ~25 positions/game ≈ 50M unique → ~5GB — fits comfortably on a
    # 32GB box. Pass --no-dedup to disable if memory is tight.
    seen_zobrist: set[int] | None = set() if dedup_positions else None

    t0 = time.time()
    print(f"Reading CSV in chunks of {chunk_size:,} (workers={num_workers})...")
    print(f"Sampling: skip_opening={skip_opening_plies}, skip_end={skip_end_plies}, "
          f"positions_per_game={positions_per_game if positions_per_game > 0 else 'all'}, "
          f"dedup={'on' if dedup_positions else 'off'}")

    init_args = (skip_opening_plies, skip_end_plies, positions_per_game, seed)

    with _open_csv_reader(csv_path, chunk_size) as reader:
        task_iter = _iter_csv_tasks(
            reader, stats, seen_urls,
            min_elo=min_elo, min_base_s=min_base_s, allow_daily=allow_daily,
        )

        if num_workers <= 1:
            # Sequential path still needs the worker config populated.
            _init_worker(*init_args)
            result_stream = (_encode_game_worker(t) for t in task_iter)
            stopped_early = _consume_results(
                result_stream, buf, stats,
                shard_size=shard_size, max_positions=max_positions, t0=t0,
                seen_zobrist=seen_zobrist,
            )
        else:
            # Windows: `spawn` is the only reliable start method; explicit here
            # for clarity / portability with fork-default Linux.
            mp_ctx = mp.get_context('spawn')
            with mp_ctx.Pool(
                processes=num_workers,
                initializer=_init_worker,
                initargs=init_args,
            ) as pool:
                result_stream = pool.imap_unordered(
                    _encode_game_worker, task_iter, chunksize=pool_chunksize,
                )
                stopped_early = _consume_results(
                    result_stream, buf, stats,
                    shard_size=shard_size, max_positions=max_positions, t0=t0,
                    seen_zobrist=seen_zobrist,
                )
                if stopped_early:
                    pool.terminate()

    written = buf.flush()
    if written:
        print(f"  wrote {written.name} (final)")

    elapsed = time.time() - t0
    print("\n=== Ingest summary ===")
    if stopped_early:
        print("  (stopped early via --max-positions — accounting check skipped)")
    else:
        stats.report(elapsed)
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Build supervised-pretraining shards from the chess.com GM dataset."
    )
    parser.add_argument('--csv', type=Path, required=True,
                        help='Local path to the chess.com games CSV '
                             '(or a .zip containing one).')
    parser.add_argument('--out-dir', type=Path,
                        default=Path('pretrain_data/chesscom'),
                        help='Shard output directory')
    parser.add_argument('--min-elo', type=int, default=0,
                        help='Minimum Elo for both players (default 0 = no filter; '
                             'the dataset is GM-only on one side, opponent rating '
                             'varies — bump to 2400 if opponent blunders hurt)')
    parser.add_argument('--min-base-s', type=int, default=180,
                        help='Minimum base time in seconds (default 180 = drop bullet; '
                             'set 0 to keep bullet, 480 to keep only rapid+)')
    parser.add_argument('--no-daily', action='store_true',
                        help='Drop correspondence/daily games (default: kept)')
    parser.add_argument('--shard-size', type=int, default=100_000,
                        help='Positions per output shard')
    parser.add_argument('--chunk-size', type=int, default=50_000,
                        help='CSV rows per pandas chunk')
    parser.add_argument('--max-positions', type=int, default=None,
                        help='Stop after this many positions (for smoke tests)')
    parser.add_argument('--start-shard', type=int, default=0,
                        help='First shard index (use when appending to existing output)')
    parser.add_argument('--workers', type=int,
                        default=max(1, (os.cpu_count() or 2) - 1),
                        help='Worker processes (default: cpu_count - 1; set 1 for sequential)')
    parser.add_argument('--pool-chunksize', type=int, default=32,
                        help='imap_unordered chunk size for IPC batching')
    parser.add_argument('--skip-opening-plies', type=int, default=12,
                        help='Drop the first N plies of each game (Maia uses 10, '
                             'Stockfish NNUE uses 16). Kills opening-theory duplication.')
    parser.add_argument('--skip-end-plies', type=int, default=5,
                        help='Drop the last N plies of each game. Cleans up noisy '
                             'policy labels from resignations / time-trouble blunders.')
    parser.add_argument('--positions-per-game', type=int, default=25,
                        help='Uniform random subsample K positions per game from the '
                             'eligible window. AlphaZero used ≤30 to decorrelate samples. '
                             'Set 0 to keep all eligible positions.')
    parser.add_argument('--no-dedup', action='store_true',
                        help='Disable corpus-wide Zobrist dedup. Dedup keeps a set of '
                             '64-bit hashes in RAM (~100B/entry); disable if memory is tight.')
    parser.add_argument('--seed', type=int, default=20260420,
                        help='RNG seed base for per-game subsampling (each worker adds its PID)')
    args = parser.parse_args()

    csv_path = args.csv
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    print(f"Using CSV: {csv_path}")

    build_shards_from_chesscom(
        csv_path=csv_path,
        out_dir=args.out_dir,
        min_elo=args.min_elo,
        min_base_s=args.min_base_s,
        allow_daily=not args.no_daily,
        shard_size=args.shard_size,
        chunk_size=args.chunk_size,
        max_positions=args.max_positions,
        start_shard=args.start_shard,
        num_workers=args.workers,
        pool_chunksize=args.pool_chunksize,
        skip_opening_plies=args.skip_opening_plies,
        skip_end_plies=args.skip_end_plies,
        positions_per_game=args.positions_per_game,
        dedup_positions=not args.no_dedup,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
