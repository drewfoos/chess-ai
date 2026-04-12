"""Build an opening-book FEN file from a PGN dataset.

For each game, extracts a position at a random ply in [min_ply, max_ply] (only
white-to-move by default, since Lc0-style training balances sides via board
flipping anyway). Dedupes by FEN. Writes one FEN per line.

Recommended PGN sources:
  - Lichess Elite database: https://database.nikonoel.fr/
  - TCEC games:             https://tcec-chess.com/
  - CCRL:                   https://www.computerchess.org.uk/ccrl/

Usage:
    python scripts/build_opening_book.py --pgn lichess_elite_2024-01.pgn \\
        --output opening_book.txt --target 50000 --min-ply 8 --max-ply 16
"""
import argparse
import random
import sys

import chess
import chess.pgn


def build(pgn_path: str, output_path: str, target: int, min_ply: int, max_ply: int,
          only_white_to_move: bool, seed: int) -> None:
    random.seed(seed)
    fens: set[str] = set()

    with open(pgn_path, encoding='utf-8', errors='replace') as f:
        games_seen = 0
        while len(fens) < target:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            games_seen += 1
            board = game.board()
            moves = list(game.mainline_moves())
            if len(moves) < min_ply:
                continue
            target_ply = random.randint(min_ply, min(max_ply, len(moves)))
            for i in range(target_ply):
                board.push(moves[i])
            if only_white_to_move and not board.turn:
                continue
            # Strip halfmove/fullmove counters for dedup (we want unique positions, not unique histories)
            parts = board.fen().split()
            key = ' '.join(parts[:4])
            if key in fens:
                continue
            fens.add(key)
            if games_seen % 1000 == 0:
                print(f"  Seen {games_seen} games, collected {len(fens)} FENs", file=sys.stderr)

    with open(output_path, 'w', encoding='utf-8') as f:
        for fen in fens:
            # Write with halfmove/fullmove = "0 1" (fresh counters)
            f.write(f"{fen} 0 1\n")
    print(f"Wrote {len(fens)} FENs to {output_path} (from {games_seen} games)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--pgn', required=True, help='Input PGN file')
    p.add_argument('--output', required=True, help='Output FEN file')
    p.add_argument('--target', type=int, default=50000, help='Target number of unique FENs')
    p.add_argument('--min-ply', type=int, default=8, help='Earliest ply to sample from')
    p.add_argument('--max-ply', type=int, default=16, help='Latest ply to sample from')
    p.add_argument('--both-sides', action='store_true', help='Include black-to-move positions too')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    build(args.pgn, args.output, args.target, args.min_ply, args.max_ply,
          only_white_to_move=not args.both_sides, seed=args.seed)


if __name__ == '__main__':
    main()
