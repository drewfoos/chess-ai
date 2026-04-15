"""Metrics logging for self-play training visualization.

Writes per-generation JSON files and a rolling summary for the dashboard.
"""

import json
import os
import tempfile
from dataclasses import dataclass, field, asdict


@dataclass
class GameMetrics:
    game_num: int
    num_moves: int
    result: str
    duration_s: float
    moves_uci: list[str] = field(default_factory=list)


@dataclass
class TrainingMetrics:
    total_loss: float
    policy_loss: float
    value_loss: float
    num_batches: int
    learning_rate: float
    # Auxiliary KataGo-style soft-policy CE, tracked separately so dashboards
    # don't conflate it with the hard policy CE in `policy_loss`.
    soft_policy_loss: float = 0.0


class MetricsLogger:
    """Writes per-generation JSON metrics for the visualization dashboard.

    Per-generation files (gen_NNN.json) are always written in full.
    summary.json keeps only the most recent `max_summary_generations`
    entries to avoid unbounded growth during long training runs.
    """

    def __init__(self, metrics_dir: str, max_summary_generations: int = 500):
        self.metrics_dir = metrics_dir
        self.max_summary_generations = max_summary_generations
        os.makedirs(metrics_dir, exist_ok=True)
        self.current_games: list[GameMetrics] = []

    def record_game(self, game: GameMetrics):
        """Record a completed game for the current generation."""
        self.current_games.append(game)

    def save_generation(
        self,
        generation: int,
        num_positions: int,
        training: TrainingMetrics,
        duration_s: float,
        network: dict | None = None,
        resumed: bool = False,
        resign_w: float | None = None,
        resign_fp_rate: float | None = None,
        discard_pool_size: int | None = None,
        adjudication_rate: float | None = None,
    ):
        """Save generation metrics to JSON and update summary.

        `resumed` marks the first generation after a training restart so the
        dashboard can annotate the loss chart with a vertical line.

        Stage 10 (Lc0-parity) adds four optional operational metrics:
        current resign threshold, its false-positive rate measured on the
        most recent playthrough games, current DiscardPool size, and the
        fraction of games adjudicated by the ply cap. Omitted fields are
        skipped so pre-Stage-10 callers keep the old JSON shape.
        """
        gen_data = {
            'generation': generation,
            'num_positions': num_positions,
            'num_games': len(self.current_games),
            'games': [asdict(g) for g in self.current_games],
            'training': asdict(training),
            'duration_s': duration_s,
        }
        if network is not None:
            gen_data['network'] = network
        if resumed:
            gen_data['resumed'] = True
        if resign_w is not None:
            gen_data['resign_w'] = resign_w
        if resign_fp_rate is not None:
            gen_data['resign_fp_rate'] = resign_fp_rate
        if discard_pool_size is not None:
            gen_data['discard_pool_size'] = discard_pool_size
        if adjudication_rate is not None:
            gen_data['adjudication_rate'] = adjudication_rate

        gen_path = os.path.join(self.metrics_dir, f'gen_{generation:03d}.json')
        with open(gen_path, 'w') as f:
            json.dump(gen_data, f, indent=2)

        self._update_summary(gen_data)
        self.current_games = []

    def _update_summary(self, gen_data: dict):
        """Append generation to summary.json."""
        summary_path = os.path.join(self.metrics_dir, 'summary.json')
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                summary = json.load(f)
        else:
            summary = {'total_generations': 0, 'generations': []}

        gen_summary = {
            'generation': gen_data['generation'],
            'num_positions': gen_data['num_positions'],
            'num_games': gen_data['num_games'],
            'training': gen_data['training'],
            'duration_s': gen_data['duration_s'],
            'network': gen_data.get('network'),
            'resumed': gen_data.get('resumed', False),
            'avg_game_length': (
                sum(g['num_moves'] for g in gen_data['games']) / max(len(gen_data['games']), 1)
            ),
            'results': {
                '1-0': sum(1 for g in gen_data['games'] if g['result'] == '1-0'),
                '0-1': sum(1 for g in gen_data['games'] if g['result'] == '0-1'),
                '1/2-1/2': sum(1 for g in gen_data['games'] if g['result'] == '1/2-1/2'),
            },
        }
        for key in ('resign_w', 'resign_fp_rate', 'discard_pool_size', 'adjudication_rate'):
            if key in gen_data:
                gen_summary[key] = gen_data[key]

        summary['generations'].append(gen_summary)

        # Trim old entries to keep summary.json bounded
        if len(summary['generations']) > self.max_summary_generations:
            summary['generations'] = summary['generations'][-self.max_summary_generations:]

        # Highest generation number we've ever seen. Using max() (instead of
        # just the current gen) makes the counter monotonic — if a run is
        # accidentally restarted at gen 1 against an existing metrics dir,
        # we keep reporting the true peak instead of resetting the dashboard.
        summary['total_generations'] = max(
            summary.get('total_generations', 0),
            gen_data['generation'],
        )

        total_positions = sum(g['num_positions'] for g in summary['generations'])
        total_time = sum(g['duration_s'] for g in summary['generations'])
        total_games = sum(g['num_games'] for g in summary['generations'])
        summary['total_positions'] = total_positions
        summary['total_games'] = total_games
        summary['total_time_s'] = total_time

        # Atomic write: write to temp file then rename to avoid corrupt reads
        fd, tmp_path = tempfile.mkstemp(dir=self.metrics_dir, suffix='.json')
        try:
            with os.fdopen(fd, 'w') as f:
                json.dump(summary, f, indent=2)
            os.replace(tmp_path, summary_path)
        except BaseException:
            os.unlink(tmp_path)
            raise
