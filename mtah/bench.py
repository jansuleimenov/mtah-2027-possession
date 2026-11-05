"""External leaderboard: leave-one-game-out on degraded Metrica games, scored by the official
possession_metrics.py. This is the only place hyperparameters are chosen.

    python -m mtah.bench                 # baseline vs model, default decode params
    python -m mtah.bench --grid          # also sweep the decoder costs
"""

import argparse
import itertools
import os
import sys

import numpy as np

from mtah.common import OFFICIAL, D, H, A, X
from mtah import external, features, model, noise
import possession_metrics as pm

sys.path.insert(0, os.path.join(OFFICIAL, "starter"))
from baseline import smooth  # noqa: E402  the official baseline's smoothing, reused as-is

TRAIN_SEEDS = (1, 2)   # two degraded copies of each training game
TEST_SEED = 100


def baseline_codes(match, window=50):
    """The published baseline on a Match: nearest player's team where the ball is seen, else D,
    then the official majority-vote smoothing. Outside the halves: X."""
    dist = np.linalg.norm(match.players - match.ball[:, None, :], axis=2)
    dist = np.where(np.isnan(dist), np.inf, dist)
    nearest = match.team[np.argmin(dist, axis=1)].astype(np.int8)
    nearest[np.isnan(match.ball[:, 0]) | ~np.isfinite(dist.min(axis=1))] = -1
    labels = pm.label_codes(smooth(nearest, window))
    labels[match.period == 0] = X
    return labels


def score(gt, pred):
    r = pm.evaluate(gt, pred)
    return r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"]


def folds():
    """Yield (test_name, test_match_degraded, [(feats, frames)], [labels]) per held-out game."""
    clean = {g: external.load(g) for g in external.GAMES}
    for test in external.GAMES:
        train_feats, train_labels = [], []
        for g in external.GAMES:
            if g == test:
                continue
            for seed in TRAIN_SEEDS:
                m = noise.degrade(clean[g], np.random.default_rng(seed))
                train_feats.append(features.build(m))
                train_labels.append(m.labels)
        held = noise.degrade(clean[test], np.random.default_rng(TEST_SEED))
        yield test, held, train_feats, train_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", action="store_true")
    args = parser.parse_args()

    grid = [model.DecodeParams()]
    if args.grid:
        grid = [model.DecodeParams(s, t, f) for s, t, f in
                itertools.product((5, 10, 20, 40), (5, 10, 20), (5, 10, 20))]

    results = {i: [] for i in range(len(grid))}
    base = []
    for test, held, train_feats, train_labels in folds():
        graded = held.labels != X
        print("{}: ball missing on graded {:.1%}".format(test, np.isnan(held.ball[graded, 0]).mean()))
        base.append(score(held.labels, baseline_codes(held)))
        clf = model.fit(train_feats, train_labels)
        feats, frames = features.build(held)
        prob = model.proba(clf, feats)
        for i, params in enumerate(grid):
            pred = model.decode(prob, frames, held.half_runs(), held.n_frames, params)
            results[i].append(score(held.labels, pred))
        print("  baseline spellF1 {:.4f} macroF1 {:.4f} spells {}/{}".format(*base[-1]))
        print("  model    spellF1 {:.4f} macroF1 {:.4f} spells {}/{}".format(*results[0][-1]))

    print("\nmean over {} folds".format(len(base)))
    print("  baseline_smooth50  spellF1 {:.4f}  macroF1 {:.4f}".format(
        np.mean([b[0] for b in base]), np.mean([b[1] for b in base])))
    ranked = sorted(results, key=lambda i: -np.mean([r[0] for r in results[i]]))
    for i in ranked[:10]:
        print("  {}  spellF1 {:.4f}  macroF1 {:.4f}".format(
            grid[i], np.mean([r[0] for r in results[i]]), np.mean([r[1] for r in results[i]])))


if __name__ == "__main__":
    main()
