"""Parallel orchestrator for Phase B Stockfish distillation.

Launches N `training.stockfish_label` subprocesses in parallel, each assigned
a stride/offset partition of the PGN game stream (worker i handles games
where `idx % N == i`). Each worker writes to its own `part_{i}` subdirectory
so shard filenames never collide, and progress is resumable — on ctrl-c,
re-running picks up where each worker left off by counting existing shards.

Why round-robin instead of contiguous game ranges: contiguous ranges force
worker N-1 to skip past `(N-1) * games_per_worker` games before it can start
labeling. With `chess.pgn.read_game` parsing every game body, that skip costs
hours for large N. Round-robin lets each worker fast-skip non-matching games
via `chess.pgn.skip_game` (~100k games/s, body-only), so no worker pays a
serial skip overhead.

Example:
    python scripts/phase_b_parallel.py \\
        --pgn lichess_db_standard_rated_2026-03.pgn \\
        --out-dir pretrain_data/phase_b \\
        --stockfish engines/stockfish/stockfish-windows-x86-64-avx2.exe \\
        --workers 16 --threads 1 --depth 12 --multipv 10 \\
        --min-elo 1800 --positions-per-worker 62500
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def count_positions_in_dir(dir_path: Path, shard_size: int) -> int:
    """Estimate positions already labeled by counting shard_*.npz files.

    Underestimates by up to `shard_size` positions (the in-flight buffer that
    hadn't flushed when the previous run exited), but that's fine — any lost
    positions get re-labeled on resume, which just means minor duplication at
    the worker-boundary seams. Not worth a state file for that.
    """
    if not dir_path.is_dir():
        return 0
    shards = list(dir_path.glob('shard_*.npz'))
    return len(shards) * shard_size


def build_worker_cmd(
    worker_id: int,
    pgn: str,
    out_dir: Path,
    stockfish: str,
    threads: int,
    depth: int,
    multipv: int,
    min_elo: int,
    temperature_cp: float,
    shard_size: int,
    positions_per_game: int,
    skip_opening_plies: int,
    hash_mb: int,
    stride: int,
    offset: int,
    skip_matched_games: int,
    max_positions: int,
    start_shard: int,
) -> list[str]:
    # -u disables Python stdout/stderr buffering so print() calls in the
    # labeler flush immediately to the redirected worker log; otherwise the
    # log stays empty for 200+s at a time (block-buffered when not a tty).
    return [
        sys.executable, '-u', '-m', 'training.stockfish_label',
        '--pgn', pgn,
        '--out-dir', str(out_dir),
        '--stockfish', stockfish,
        '--threads', str(threads),
        '--depth', str(depth),
        '--multipv', str(multipv),
        '--min-elo', str(min_elo),
        '--temperature-cp', str(temperature_cp),
        '--shard-size', str(shard_size),
        '--positions-per-game', str(positions_per_game),
        '--skip-opening-plies', str(skip_opening_plies),
        '--hash-mb', str(hash_mb),
        '--stride', str(stride),
        '--offset', str(offset),
        '--skip-matched-games', str(skip_matched_games),
        '--max-positions', str(max_positions),
        '--start-shard', str(start_shard),
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description='Parallel Phase B Stockfish distillation')
    ap.add_argument('--pgn', required=True)
    ap.add_argument('--out-dir', type=Path, required=True,
                    help='Parent dir; each worker writes to part_{i} subdir')
    ap.add_argument('--stockfish', default='engines/stockfish/stockfish-windows-x86-64-avx2.exe')
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--threads', type=int, default=1,
                    help='SF threads per worker. On Ryzen 5800X, 1 thread × 16 workers '
                         'is ~2x faster than 2 × 8 for multipv=10 searches.')
    ap.add_argument('--depth', type=int, default=13,
                    help='SF search depth. 13 gives ~2800 Elo labels at 6h/2M on this HW; '
                         '12 is ~2600 Elo at 4h/2M. 15 is overkill for supervised labels.')
    ap.add_argument('--multipv', type=int, default=10)
    ap.add_argument('--min-elo', type=int, default=2000)
    ap.add_argument('--temperature-cp', type=float, default=80.0)
    ap.add_argument('--shard-size', type=int, default=50_000)
    ap.add_argument('--positions-per-game', type=int, default=3)
    ap.add_argument('--skip-opening-plies', type=int, default=10)
    ap.add_argument('--hash-mb', type=int, default=256)
    ap.add_argument('--positions-per-worker', type=int, default=125_000,
                    help='Per-worker position cap (e.g. 2M total / 16 workers default).')
    args = ap.parse_args()

    if not os.path.isfile(args.stockfish):
        print(f"Stockfish binary not found: {args.stockfish}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Round-robin partitioning: each worker opens its own PGN reader and
    # filters games by `idx % stride == offset`, fast-skipping non-matching
    # games via `chess.pgn.skip_game`. No worker pays a large skip offset.
    procs: list[subprocess.Popen] = []
    for i in range(args.workers):
        part_dir = args.out_dir / f'part_{i:02d}'
        already_done = count_positions_in_dir(part_dir, args.shard_size)
        existing_shards = already_done // args.shard_size
        if already_done >= args.positions_per_worker:
            print(f"[worker {i:02d}] already complete ({already_done:,} pos) — skipping")
            procs.append(None)  # type: ignore[arg-type]
            continue

        # Resume: skip past the matched games already labeled in a prior run.
        # Approximated by positions / positions_per_game; worst case re-labels
        # at most one shard's worth at the seam (acceptable, see docstring
        # in count_positions_in_dir).
        skip_matched = already_done // max(args.positions_per_game, 1)
        remaining_positions = args.positions_per_worker - already_done

        cmd = build_worker_cmd(
            worker_id=i,
            pgn=args.pgn,
            out_dir=part_dir,
            stockfish=args.stockfish,
            threads=args.threads,
            depth=args.depth,
            multipv=args.multipv,
            min_elo=args.min_elo,
            temperature_cp=args.temperature_cp,
            shard_size=args.shard_size,
            positions_per_game=args.positions_per_game,
            skip_opening_plies=args.skip_opening_plies,
            hash_mb=args.hash_mb,
            stride=args.workers,
            offset=i,
            skip_matched_games=skip_matched,
            max_positions=remaining_positions,
            start_shard=existing_shards,
        )

        log_path = args.out_dir / f'worker_{i:02d}.log'
        log_fh = open(log_path, 'a', encoding='utf-8')
        log_fh.write(f"\n=== launched {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        log_fh.write(f"cmd: {' '.join(cmd)}\n")
        log_fh.flush()
        proc = subprocess.Popen(
            cmd, stdout=log_fh, stderr=subprocess.STDOUT,
        )
        procs.append(proc)
        print(f"[worker {i:02d}] stride={args.workers} offset={i} "
              f"resume_from_shard={existing_shards} skip_matched={skip_matched:,} "
              f"remaining={remaining_positions:,} pos -> {log_path.name}")

    if not any(p for p in procs):
        print("All workers already complete — nothing to do.")
        return 0

    # Propagate ctrl-c to all workers cleanly.
    def _shutdown(signum, frame):
        print(f"\nReceived signal {signum}, terminating workers...")
        for p in procs:
            if p is not None and p.poll() is None:
                p.terminate()
        for p in procs:
            if p is not None:
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill()
        sys.exit(130)

    signal.signal(signal.SIGINT, _shutdown)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, _shutdown)

    # Poll loop: print per-worker position totals every 30s.
    t_start = time.time()
    try:
        while True:
            alive = [p for p in procs if p is not None and p.poll() is None]
            totals = [count_positions_in_dir(args.out_dir / f'part_{i:02d}', args.shard_size)
                      for i in range(args.workers)]
            grand = sum(totals)
            elapsed = time.time() - t_start
            eta_h = (args.positions_per_worker * args.workers - grand) / max(grand / max(elapsed, 1), 1) / 3600 if grand > 0 else float('inf')
            print(f"[{int(elapsed):>6}s] alive={len(alive)}/{args.workers} "
                  f"total≈{grand:,} pos  ({grand/max(elapsed,1):.1f} pos/s)  "
                  f"ETA {eta_h:.1f}h")
            if not alive:
                break
            time.sleep(30)
    except KeyboardInterrupt:
        _shutdown(signal.SIGINT, None)

    failed = [i for i, p in enumerate(procs) if p is not None and p.returncode not in (0, None)]
    if failed:
        print(f"\n{len(failed)} workers exited with errors: {failed}")
        print("See worker_XX.log files in the output dir.")
        return 1

    total_positions = sum(
        count_positions_in_dir(args.out_dir / f'part_{i:02d}', args.shard_size)
        for i in range(args.workers)
    )
    print(f"\nDone. ~{total_positions:,} positions labeled across {args.workers} workers "
          f"in {(time.time() - t_start) / 3600:.2f}h")
    return 0


if __name__ == '__main__':
    sys.exit(main())
