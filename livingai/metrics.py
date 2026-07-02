"""In-memory metrics counters for the Living AI runtime.

Deliberately tiny and dependency-free. Production deployments can swap this for a
Prometheus/StatsD adapter; the runtime only relies on the ``increment`` and
``observe`` methods.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any


__all__ = ["Metrics"]


class Metrics:
    """Thread-safe counters and value observations.

    * ``increment(name)`` bumps a monotonic counter.
    * ``observe(name, value)`` records a sample (e.g. a latency in ms) for later
      percentile analysis.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, int] = defaultdict(int)
        self._samples: dict[str, list[float]] = defaultdict(list)

    def increment(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._counters[name] += amount

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            self._samples[name].append(value)

    def counter(self, name: str) -> int:
        with self._lock:
            return self._counters[name]

    def samples(self, name: str) -> list[float]:
        with self._lock:
            return list(self._samples[name])

    def percentile(self, name: str, pct: float) -> float | None:
        """Return the ``pct`` percentile (0-100) of observed samples, or None."""
        with self._lock:
            data = sorted(self._samples[name])
        if not data:
            return None
        if len(data) == 1:
            return data[0]
        rank = (pct / 100.0) * (len(data) - 1)
        low = int(rank)
        high = min(low + 1, len(data) - 1)
        frac = rank - low
        return data[low] + (data[high] - data[low]) * frac

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "counters": dict(self._counters),
                "samples": {k: list(v) for k, v in self._samples.items()},
            }
