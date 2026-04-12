"""One-shot converter: legacy dense `planes` npz → packed v2 (bitboards + metadata).

Scans a directory (default `models/current_run/data`) for `gen_*.npz` files
written in the legacy dense format (`planes` key, shape (N,112,8,8) float32)
and rewrites each in place as a gzipped packed-v2 archive. The conversion is
lossless for the 104 binary piece/history planes (0-103) and exact for the
scalar feature planes (104-110) up to the representational precision of the
metadata (uint8/uint16).

Usage:
    python -m scripts.convert_npz_to_bitboards \
        --data-dir models/current_run/data --backup --verify
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Reuse the exact same packers used at load time so converter and dataset
# can't drift apart.
from training.dataset import _pack_dense_planes, _extract_metadata_from_dense, unpack_bitboards


def _is_packed(data: np.lib.npyio.NpzFile) -> bool:
    return 'bitboards' in data.files


def _convert_one(src_path: Path, out_path: Path, verify: bool) -> dict:
    """Convert one file. Returns a small report dict for logging."""
    with np.load(src_path) as data:
        if _is_packed(data):
            return {'path': str(src_path), 'status': 'already_packed', 'skipped': True}

        planes = np.asarray(data['planes'], dtype=np.float32)
        policies = np.asarray(data['policies'], dtype=np.float32)
        values = np.asarray(data['values'], dtype=np.float32)
        extras = {}
        for k in ('moves_left', 'surprise', 'use_policy'):
            if k in data.files:
                extras[k] = np.asarray(data[k])

        meta = _extract_metadata_from_dense(planes)
        bitboards = _pack_dense_planes(planes)

        if verify:
            # Sanity-check the round trip on the first few rows.
            n_check = min(4, planes.shape[0])
            expanded = unpack_bitboards(bitboards[:n_check])  # (n, 104, 8, 8)
            bin_planes = (planes[:n_check, :104] > 0.5).astype(np.float32)
            if not np.array_equal(expanded, bin_planes):
                raise RuntimeError(
                    f"{src_path}: bitboard round-trip mismatch on binary planes"
                )

    np.savez_compressed(
        out_path,
        format_version=np.uint8(2),
        bitboards=bitboards,
        stm=meta['stm'],
        castling=meta['castling'],
        rule50=meta['rule50'],
        fullmove=meta['fullmove'],
        policies=policies,
        values=values,
        **extras,
    )
    return {
        'path': str(src_path),
        'status': 'converted',
        'num_positions': int(planes.shape[0]),
        'src_bytes': src_path.stat().st_size,
        'out_bytes': out_path.stat().st_size,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', default='models/current_run/data',
                   help='Directory containing gen_*.npz files.')
    p.add_argument('--pattern', default='gen_*.npz', help='Glob pattern to match.')
    p.add_argument('--backup', action='store_true',
                   help='Keep a .bak copy of the original next to each file.')
    p.add_argument('--verify', action='store_true',
                   help='Round-trip verify a few rows of each converted file before committing.')
    p.add_argument('--dry-run', action='store_true',
                   help='Print plan but do not write.')
    args = p.parse_args(argv)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"error: data directory not found: {data_dir}", file=sys.stderr)
        return 2

    sources = sorted(data_dir.glob(args.pattern))
    if not sources:
        print(f"no files matched {data_dir / args.pattern}")
        return 0

    print(f"Found {len(sources)} file(s) in {data_dir}")
    total_src = 0
    total_out = 0
    converted = 0
    for src in sources:
        with np.load(src) as d:
            if _is_packed(d):
                print(f"  skip (already packed): {src.name}")
                continue

        if args.dry_run:
            print(f"  would convert: {src.name}")
            continue

        # Keep the .npz extension on the tmp file — np.savez_compressed appends
        # .npz to any path missing it, which would break the rename below.
        tmp = src.with_name(src.stem + '.tmp.npz')
        report = _convert_one(src, tmp, args.verify)
        if report.get('skipped'):
            tmp.unlink(missing_ok=True)
            continue

        # Atomic-ish swap: back up then rename tmp → src.
        if args.backup:
            bak = src.with_suffix(src.suffix + '.bak')
            if bak.exists():
                bak.unlink()
            src.rename(bak)
        else:
            src.unlink()
        tmp.rename(src)

        converted += 1
        total_src += report['src_bytes']
        total_out += report['out_bytes']
        ratio = report['out_bytes'] / report['src_bytes'] if report['src_bytes'] else 0.0
        print(f"  converted {src.name}: {report['num_positions']} positions, "
              f"{report['src_bytes'] / 1e6:.1f}MB → {report['out_bytes'] / 1e6:.1f}MB "
              f"({ratio:.2%})")

    if converted:
        overall = total_out / total_src if total_src else 0.0
        print(f"Done: {converted} files, {total_src / 1e6:.1f}MB → {total_out / 1e6:.1f}MB "
              f"({overall:.2%} of original)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
