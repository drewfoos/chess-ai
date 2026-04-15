"""Count actual positions across all phase_b shards.

Monitor in phase_b_parallel.py counts shard files × shard_size, which
overcounts whenever a worker exits with a partial in-flight shard. This
script opens each .npz and sums the real policy-index row counts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', type=Path, default=Path('pretrain_data/phase_b'))
    args = ap.parse_args()

    total = 0
    per_worker: dict[str, int] = {}
    first = True
    for p in sorted(args.dir.rglob('shard_*.npz')):
        d = np.load(p)
        if first:
            print(f'Keys in {p.name}: {list(d.keys())}')
            first = False
        n = len(d['bitboards'])
        total += n
        per_worker[p.parent.name] = per_worker.get(p.parent.name, 0) + n
        print(f'{p.parent.name}/{p.name}: {n:,}')

    print()
    print('Per worker:')
    for name in sorted(per_worker):
        print(f'  {name}: {per_worker[name]:,}')
    print(f'\nTOTAL: {total:,} positions across {len(per_worker)} workers')


if __name__ == '__main__':
    main()
