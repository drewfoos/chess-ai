"""Generate synthetic training data for testing the training pipeline.

This produces random positions with random policy and value targets.
Real training data comes from self-play (Plan 4).
"""

import argparse
import numpy as np
from training.encoder import encode_position, POLICY_SIZE


# Standard FENs for generating diverse synthetic positions
_SEED_FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
    "r1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",
    "rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "rnbqkb1r/pp2pppp/5n2/2pp4/4P3/2N5/PPPP1PPP/R1BQKBNR w KQkq d6 0 3",
]


def generate_synthetic_data(output_path: str, num_positions: int = 1000):
    """Generate synthetic training data and save as .npz file.

    Each sample has:
    - planes: encoded position (112, 8, 8)
    - policy: random probability distribution over 1858 moves
    - value: random WDL target (one of [1,0,0], [0,1,0], [0,0,1])
    """
    rng = np.random.default_rng(42)

    planes_list = []
    policies_list = []
    values_list = []

    for i in range(num_positions):
        # Cycle through seed FENs
        fen = _SEED_FENS[i % len(_SEED_FENS)]
        planes = encode_position(fen)
        planes_list.append(planes)

        # Random policy: Dirichlet-distributed over a random subset of moves
        policy = rng.dirichlet(np.ones(POLICY_SIZE) * 0.03)
        policies_list.append(policy.astype(np.float32))

        # Random WDL outcome
        outcome = rng.choice(3)
        value = np.zeros(3, dtype=np.float32)
        value[outcome] = 1.0
        values_list.append(value)

    np.savez(
        output_path,
        planes=np.array(planes_list, dtype=np.float32),
        policies=np.array(policies_list, dtype=np.float32),
        values=np.array(values_list, dtype=np.float32),
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate synthetic training data')
    parser.add_argument('--output', type=str, default='data/synthetic.npz')
    parser.add_argument('--num-positions', type=int, default=1000)
    args = parser.parse_args()

    import os
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    generate_synthetic_data(args.output, args.num_positions)
    print(f"Generated {args.num_positions} positions -> {args.output}")
