"""Summary of the external games: label mix, possession counts, ball visibility by state."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from mtah.common import H, A, D, X  # noqa: E402  (puts the official scorer on sys.path)
from mtah import external  # noqa: E402
import possession_metrics as pm  # noqa: E402

for game in external.GAMES:
    m = external.load(game)
    runs = pm.valid_runs(m.labels)
    stats = pm.possession_stats(m.labels, runs)
    graded = m.labels != X
    ball_seen = ~np.isnan(m.ball[:, 0])
    print("{}: frames {} graded {} | H {} A {} D {} | spells {} mean {:.1f}s | slots {}".format(
        m.name, m.n_frames, graded.sum(), (m.labels == H).sum(), (m.labels == A).sum(),
        (m.labels == D).sum(), stats["n_spells"], stats["mean_duration_s"], len(m.team)))
    for name, code in (("alive", None), ("dead", D)):
        mask = graded & ((m.labels != D) if code is None else (m.labels == D))
        print("   ball visible when {}: {:.1%}".format(name, ball_seen[mask].mean()))
    print("   halves", m.half_runs())
