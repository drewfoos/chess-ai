"""Bounded FIFO of rejected positions, shared across games and persisted.

Self-play produces positions the temperature sampler picks but then rejects
via the min-visit floor (Stage 5). Instead of throwing them away, we keep a
bounded pool of such positions and let Stage 6 seed a fraction of new games
from the pool — turning noise into targeted exploration.

Persistence is a JSON file written atomically by the training_loop hook after
each generation; load on the next run picks up where we left off. Thread-safe
so the concurrent self-play workers can push without coordinating.
"""
from __future__ import annotations

import json
import threading
from collections import deque
from pathlib import Path


class DiscardPool:
    def __init__(self, cap: int, persist_path: Path | str | None = None):
        self.cap = cap
        self.persist_path = Path(persist_path) if persist_path else None
        self._dq: deque[str] = deque(maxlen=cap)
        self._lock = threading.Lock()

    def push(self, fen: str) -> None:
        with self._lock:
            self._dq.append(fen)

    def pop(self) -> str | None:
        with self._lock:
            if not self._dq:
                return None
            return self._dq.popleft()

    def size(self) -> int:
        with self._lock:
            return len(self._dq)

    def snapshot(self) -> list[str]:
        with self._lock:
            return list(self._dq)

    def save(self) -> None:
        if not self.persist_path:
            return
        with self._lock:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            self.persist_path.write_text(json.dumps(list(self._dq)))

    def load(self) -> None:
        if not self.persist_path or not self.persist_path.exists():
            return
        data = json.loads(self.persist_path.read_text())
        with self._lock:
            self._dq.clear()
            for fen in data[-self.cap:]:
                self._dq.append(fen)
