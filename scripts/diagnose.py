"""Compare feature distributions on the competition match with the external training games.

Prints aggregates only (one median and a training range per feature, and a few counts), never a
frame-level value, so the output is safe to share.

    python scripts/diagnose.py --data C:/mtah-data
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from mtah import dataset, features  # noqa: E402
from mtah.bench import TEST_SEED  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True)
args = parser.parse_args()
os.environ["MTAH_DATA_DIR"] = args.data
from mtah import competition  # noqa: E402

m = competition.load(args.data)
feats, frames = features.build(m)

graded = m.period > 0
print("slots {} (home {} away {})".format(len(m.team), int((m.team == 0).sum()), int((m.team == 1).sum())))
print("halves {}  kickoffs {}  set-piece tags {}".format(m.half_runs(), m.meta["kickoff"],
                                                         len(m.meta["setpieces"])))
print("ball seen on graded frames {:.1%}".format(np.mean(~np.isnan(m.ball[graded, 0]))))
print("players tracked per graded frame {:.1f}".format(
    np.mean(np.sum(~np.isnan(m.players[graded, :, 0]), axis=1))))
bx = m.ball[graded, 0][~np.isnan(m.ball[graded, 0])]
px = m.players[graded, :, 0][~np.isnan(m.players[graded, :, 0])]
print("ball x range p1..p99 {:.1f}..{:.1f} m   players x p1..p99 {:.1f}..{:.1f} m   pitch {}".format(
    *np.percentile(bx, [1, 99]), *np.percentile(px, [1, 99]), m.pitch))

train = [dataset.item(s, g, load, TEST_SEED)["feats"] for s, g, load in dataset.games()]
train_med = np.array([t.median().to_numpy() for t in train])   # one median per game
lo, hi = np.percentile(train_med, 5, axis=0), np.percentile(train_med, 95, axis=0)
comp = feats.median().to_numpy()

print("\n{:<18} {:>10} {:>10} {:>10}  {}".format("feature", "match", "train p5", "train p95", ""))
for name, c, a, b in zip(feats.columns, comp, lo, hi):
    flag = "  <-- outside" if (c < a - 1e-9 or c > b + 1e-9) else ""
    print("{:<18} {:>10.2f} {:>10.2f} {:>10.2f}{}".format(name, c, a, b, flag))
