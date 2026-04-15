"""Self-play training shard format v2.

One .npz file per game (or per N-game chunk). Header fields are constant
across the shard; per-row fields are stacked along axis 0.

Compared to v1, v2 decomposes the eval signals the self-play loop already
computed (best_eval / played_eval / raw_nn_eval) instead of collapsing them
into a single Q-blended `value_target`. The trainer blends at load time via
`blend_value_target`, so experimenting with blend weights no longer requires
regenerating the shards.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

SCHEMA_VERSION = 2


@dataclass
class ShardHeader:
    final_wdl: Tuple[float, float, float]
    seed_source: str           # "opening_book" | "discard_pool" | "standard"
    net_generation: int
    was_playthrough: bool
    adjudicated: bool


@dataclass
class RecordRow:
    planes: np.ndarray         # (112, 8, 8) uint8
    visits_policy: np.ndarray  # (1858,) float32
    soft_policy: np.ndarray    # (1858,) float32
    best_eval: np.ndarray      # (3,) float32
    played_eval: np.ndarray    # (3,) float32
    raw_nn_eval: np.ndarray    # (3,) float32
    mlh: np.float32
    side_to_move: np.int8
    is_full_search: bool
    was_playthrough: bool
    adjudicated: bool


def write_shard(path: Path, header: ShardHeader, rows: List[RecordRow]) -> None:
    n = len(rows)
    np.savez_compressed(
        str(path),
        schema_version=np.int32(SCHEMA_VERSION),
        final_wdl=np.array(header.final_wdl, dtype=np.float32),
        seed_source=np.str_(header.seed_source),
        net_generation=np.int32(header.net_generation),
        header_was_playthrough=np.bool_(header.was_playthrough),
        header_adjudicated=np.bool_(header.adjudicated),
        planes=np.stack([r.planes for r in rows]) if n else np.zeros((0, 112, 8, 8), np.uint8),
        visits_policy=np.stack([r.visits_policy for r in rows]) if n else np.zeros((0, 1858), np.float32),
        soft_policy=np.stack([r.soft_policy for r in rows]) if n else np.zeros((0, 1858), np.float32),
        best_eval=np.stack([r.best_eval for r in rows]) if n else np.zeros((0, 3), np.float32),
        played_eval=np.stack([r.played_eval for r in rows]) if n else np.zeros((0, 3), np.float32),
        raw_nn_eval=np.stack([r.raw_nn_eval for r in rows]) if n else np.zeros((0, 3), np.float32),
        mlh=np.array([r.mlh for r in rows], dtype=np.float32),
        side_to_move=np.array([r.side_to_move for r in rows], dtype=np.int8),
        is_full_search=np.array([r.is_full_search for r in rows], dtype=np.bool_),
        was_playthrough=np.array([r.was_playthrough for r in rows], dtype=np.bool_),
        adjudicated=np.array([r.adjudicated for r in rows], dtype=np.bool_),
    )


def read_shard(path: Path) -> Tuple[ShardHeader, List[RecordRow]]:
    z = np.load(str(path), allow_pickle=False)
    assert int(z["schema_version"]) == SCHEMA_VERSION, f"shard {path} is not v2"
    header = ShardHeader(
        final_wdl=tuple(z["final_wdl"].tolist()),
        seed_source=str(z["seed_source"]),
        net_generation=int(z["net_generation"]),
        was_playthrough=bool(z["header_was_playthrough"]),
        adjudicated=bool(z["header_adjudicated"]),
    )
    n = z["planes"].shape[0]
    rows = [
        RecordRow(
            planes=z["planes"][i],
            visits_policy=z["visits_policy"][i],
            soft_policy=z["soft_policy"][i],
            best_eval=z["best_eval"][i],
            played_eval=z["played_eval"][i],
            raw_nn_eval=z["raw_nn_eval"][i],
            mlh=z["mlh"][i],
            side_to_move=z["side_to_move"][i],
            is_full_search=bool(z["is_full_search"][i]),
            was_playthrough=bool(z["was_playthrough"][i]),
            adjudicated=bool(z["adjudicated"][i]),
        )
        for i in range(n)
    ]
    return header, rows
