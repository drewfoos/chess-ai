"""Elo gauntlet: our engine vs. Stockfish at multiple skill levels.

Plays N games per Stockfish skill level with color alternation, reports W/D/L
and an Elo estimate per opponent. Intended for on-demand tracking of training
progress, not auto-integrated into the training loop.

Stockfish skill level → approximate Elo (from Stockfish docs):
    0 → ~1320   2 → ~1500   4 → ~1700   6 → ~1900
    8 → ~2100   10 → ~2300  15 → ~2700  20 → ~3200+ (no limit)

Usage:
    python scripts/eval_vs_stockfish.py \\
        --engine build/Release/chess_engine.exe \\
        --model models/current_run/cpp_model.trt \\
        --stockfish engines/stockfish/stockfish-windows-x86-64-avx2.exe \\
        --levels 0,2,4,6 \\
        --games 20 \\
        --sims 400
"""
import argparse
import math
import os
import sys
from dataclasses import dataclass

import chess
import chess.engine


@dataclass
class Result:
    wins: int = 0
    draws: int = 0
    losses: int = 0

    @property
    def total(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def score(self) -> float:
        return self.wins + 0.5 * self.draws


def elo_from_score(score: float, games: int) -> float:
    """Elo difference from score fraction. Returns ±800 cap at 0/1."""
    if games == 0:
        return 0.0
    p = score / games
    if p <= 0.001:
        return -800.0
    if p >= 0.999:
        return 800.0
    return -400.0 * math.log10(1.0 / p - 1.0)


def play_game(our_engine: chess.engine.SimpleEngine,
              sf_engine: chess.engine.SimpleEngine,
              our_is_white: bool,
              movetime_ms: int,
              max_plies: int) -> str:
    """Play one game. Returns '1-0', '0-1', or '1/2-1/2' from our engine's POV."""
    board = chess.Board()
    limit = chess.engine.Limit(time=movetime_ms / 1000.0)

    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        our_turn = (board.turn == chess.WHITE) == our_is_white
        engine = our_engine if our_turn else sf_engine
        try:
            result = engine.play(board, limit)
        except chess.engine.EngineError as e:
            print(f"    [engine error: {e}]", file=sys.stderr)
            break
        if result.move is None:
            break
        board.push(result.move)

    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return '1/2-1/2'
    if outcome.winner is None:
        return '1/2-1/2'
    won = (outcome.winner == chess.WHITE) == our_is_white
    return '1-0' if won else '0-1'


def play_match(our_cmd: list[str],
               sf_path: str,
               skill_level: int,
               num_games: int,
               movetime_ms: int,
               max_plies: int) -> Result:
    result = Result()
    our_engine = chess.engine.SimpleEngine.popen_uci(our_cmd)
    sf_engine = chess.engine.SimpleEngine.popen_uci(sf_path)
    try:
        sf_engine.configure({'Skill Level': skill_level})

        for g in range(num_games):
            our_is_white = (g % 2 == 0)
            # python-chess SimpleEngine has no public ucinewgame(); engines
            # reset state implicitly when play() gets a fresh board.
            outcome = play_game(our_engine, sf_engine, our_is_white, movetime_ms, max_plies)
            if outcome == '1-0':
                result.wins += 1
                mark = 'W'
            elif outcome == '0-1':
                result.losses += 1
                mark = 'L'
            else:
                result.draws += 1
                mark = 'D'
            color = 'W' if our_is_white else 'B'
            print(f"    game {g + 1}/{num_games} ({color}): {mark}  "
                  f"running {result.wins}-{result.draws}-{result.losses}")
    finally:
        our_engine.quit()
        sf_engine.quit()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--engine', default='build/Release/chess_engine.exe',
                    help='Path to our chess_engine binary')
    ap.add_argument('--model', required=True,
                    help='Path to TorchScript .pt or TensorRT .trt model')
    ap.add_argument('--device', default='cuda',
                    help='Device for TorchScript (ignored for .trt)')
    ap.add_argument('--stockfish', default='engines/stockfish/stockfish-windows-x86-64-avx2.exe',
                    help='Path to Stockfish binary')
    ap.add_argument('--levels', default='0,2,4,6',
                    help='Comma-separated Stockfish skill levels to test')
    ap.add_argument('--games', type=int, default=20,
                    help='Games per skill level (even number recommended)')
    ap.add_argument('--sims', type=int, default=400,
                    help='MCTS simulations per move for our engine')
    ap.add_argument('--movetime', type=int, default=0,
                    help='Move time in ms (0 = use sims via nodes UCI limit)')
    ap.add_argument('--max-plies', type=int, default=400,
                    help='Max half-moves before adjudicating as draw')
    args = ap.parse_args()

    if not os.path.isfile(args.engine):
        print(f"Error: engine not found: {args.engine}", file=sys.stderr)
        return 1
    if not os.path.isfile(args.model):
        print(f"Error: model not found: {args.model}", file=sys.stderr)
        return 1
    if not os.path.isfile(args.stockfish):
        print(f"Error: Stockfish not found: {args.stockfish}", file=sys.stderr)
        return 1

    is_trt = args.model.lower().endswith('.trt')
    if is_trt:
        our_cmd = [args.engine, 'uci_trt', args.model]
    else:
        our_cmd = [args.engine, 'uci', args.model, args.device]

    movetime_ms = args.movetime if args.movetime > 0 else max(200, args.sims * 2)

    levels = [int(x) for x in args.levels.split(',')]

    print(f"Our engine: {' '.join(our_cmd)}")
    print(f"Stockfish:  {args.stockfish}")
    print(f"Games/level: {args.games}   movetime: {movetime_ms}ms   max plies: {args.max_plies}")
    print(f"Skill levels: {levels}")
    print()

    results: dict[int, Result] = {}
    for level in levels:
        print(f"=== Stockfish skill level {level} ===")
        r = play_match(our_cmd, args.stockfish, level, args.games, movetime_ms, args.max_plies)
        results[level] = r
        elo = elo_from_score(r.score, r.total)
        print(f"  Result: {r.wins}W - {r.draws}D - {r.losses}L   "
              f"score {r.score}/{r.total} ({100 * r.score / r.total:.1f}%)   "
              f"elo diff: {elo:+.0f}")
        print()

    print("=" * 60)
    print(f"{'Level':>6} {'W':>4} {'D':>4} {'L':>4} {'Score':>8} {'Pct':>6} {'EloDiff':>9}")
    print("-" * 60)
    for level, r in results.items():
        elo = elo_from_score(r.score, r.total)
        pct = 100 * r.score / r.total if r.total else 0.0
        print(f"{level:>6} {r.wins:>4} {r.draws:>4} {r.losses:>4} "
              f"{r.score:>8.1f} {pct:>5.1f}% {elo:>+8.0f}")
    print("=" * 60)
    return 0


if __name__ == '__main__':
    sys.exit(main())
