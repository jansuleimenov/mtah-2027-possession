"""Shared types and constants.

A Match is the one in-memory shape every loader produces and every later stage consumes, so the
competition game and the external training games go through exactly the same code.
"""

import os
import sys
from dataclasses import dataclass, field

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OFFICIAL = os.path.join(ROOT, "vendor", "official")
EXTERNAL = os.path.join(ROOT, "external")

# The official scorer and frame alignment are used as-is, never copied.
for _path in (OFFICIAL, os.path.join(OFFICIAL, "evaluation")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# Label codes, identical to evaluation/possession_metrics.py.
H, A, D, X = 0, 1, 2, 3
LABELS = np.array(["H", "A", "D", "X"])
HOME, AWAY = 0, 1
FPS = 25


@dataclass
class Match:
    """One match on one timeline. Index i is frame i (0-based) of that match.

    ball     (T, 2) metres, NaN where the ball is not observed
    players  (T, S, 2) metres, NaN where a track slot is empty
    team     (S,) HOME or AWAY for each track slot
    period   (T,) 1 or 2 inside a half, 0 outside play (pre-match, halftime, post-match)
    labels   (T,) H/A/D/X codes where known (external games), else None
    """
    name: str
    ball: np.ndarray
    players: np.ndarray
    team: np.ndarray
    period: np.ndarray
    pitch: tuple = (105.0, 68.0)
    labels: np.ndarray = None
    meta: dict = field(default_factory=dict)

    @property
    def n_frames(self):
        return len(self.ball)

    def half_runs(self):
        """[(half, start, stop)] half-open frame spans of each period."""
        runs = []
        for half in (1, 2):
            idx = np.flatnonzero(self.period == half)
            if len(idx):
                runs.append((half, int(idx[0]), int(idx[-1]) + 1))
        return runs
