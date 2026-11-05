"""Summary of SkillCorner games, same columns as describe_external.py."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from mtah.common import H, A, D, X  # noqa: E402
from mtah import skillcorner  # noqa: E402
import possession_metrics as pm  # noqa: E402

ids = skillcorner.match_ids()
print("matches:", len(ids))
for mid in ids[: int(sys.argv[1]) if len(sys.argv) > 1 else len(ids)]:
    m = skillcorner.load(mid)
    runs = pm.valid_runs(m.labels)
    stats = pm.possession_stats(m.labels, runs)
    graded = m.labels != X
    seen = ~np.isnan(m.ball[:, 0])
    alive = graded & (m.labels != D)
    print("{}: frames {} graded {} | H {} A {} D {} | spells {} mean {:.1f}s | ball seen alive {:.1%}"
          " dead {:.1%} | players/frame {:.1f} | setpieces {}".format(
              m.name, m.n_frames, graded.sum(), (m.labels == H).sum(), (m.labels == A).sum(),
              (m.labels == D).sum(), stats["n_spells"], stats["mean_duration_s"],
              seen[alive].mean(), seen[graded & (m.labels == D)].mean(),
              np.mean(np.sum(~np.isnan(m.players[graded, :, 0]), axis=1)), len(m.meta["setpieces"])))
