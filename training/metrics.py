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
    ):
        """Save generation metrics to JSON and update summary."""
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
            'avg_game_length': (
                sum(g['num_moves'] for g in gen_data['games']) / max(len(gen_data['games']), 1)
            ),
            'results': {
                '1-0': sum(1 for g in gen_data['games'] if g['result'] == '1-0'),
                '0-1': sum(1 for g in gen_data['games'] if g['result'] == '0-1'),
                '1/2-1/2': sum(1 for g in gen_data['games'] if g['result'] == '1/2-1/2'),
            },
        }

        summary['generations'].append(gen_summary)

        # Trim old entries to keep summary.json bounded
        if len(summary['generations']) > self.max_summary_generations:
            summary['generations'] = summary['generations'][-self.max_summary_generations:]

        summary['total_generations'] = gen_data['generation']  # actual generation count, not len

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
