"""EMA-updated resign threshold calibrator.

A small fraction of games disable resign (playthrough games) so we can
measure the true distribution of "how low did the eventual winner's W go".
Taking the 95th percentile of those minima gives us a threshold that would
have false-positived on ~5% of winning games — we EMA it toward the current
resign_w each generation, with a warmup period during which the default
holds.
"""
from __future__ import annotations


class ResignCalibrator:
    def __init__(
        self,
        default: float = 0.02,
        ema_alpha: float = 0.3,
        percentile: float = 0.95,
        warmup_generations: int = 3,
    ):
        self.default = default
        self.ema_alpha = ema_alpha
        self.percentile = percentile
        self.warmup_generations = warmup_generations
        self.current = default
        self.last_fp_rate: float | None = None

    def update(self, generation: int, playthrough_min_evals: list[float]) -> float:
        """EMA-update resign_w toward the p95 of playthrough min-evals.

        During warmup (generation <= warmup_generations) the threshold stays
        locked at `default` regardless of observations. Returns the new
        threshold and also stores it as `self.current`.
        """
        if generation <= self.warmup_generations or not playthrough_min_evals:
            self.current = self.default
            return self.current
        s = sorted(playthrough_min_evals)
        idx = max(0, min(len(s) - 1, int(self.percentile * (len(s) - 1))))
        p = s[idx]
        self.current = (1 - self.ema_alpha) * self.current + self.ema_alpha * p
        return self.current

    def false_positive_rate(self, playthrough_min_evals: list[float]) -> float:
        """Fraction of playthrough games where current threshold would have resigned."""
        if not playthrough_min_evals:
            self.last_fp_rate = 0.0
            return 0.0
        fp = sum(1 for v in playthrough_min_evals if v < self.current)
        self.last_fp_rate = fp / len(playthrough_min_evals)
        return self.last_fp_rate
