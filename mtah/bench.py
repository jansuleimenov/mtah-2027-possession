"""External leaderboard, scored by the official possession_metrics.py. The only place
hyperparameters are chosen.

Three folds over all external games: fold f holds out one Metrica game and every third
SkillCorner game; the model trains on the rest. Scores are reported per source, because the two
sources define "dead ball" slightly differently and the competition's own definition is a third.

    python -m mtah.bench                 # model with the submission's decoder costs
    python -m mtah.bench --grid          # also sweep the decoder costs
    python -m mtah.bench --baseline      # also score the official baseline
"""

import argparse
import itertools
import os
import sys

import numpy as np

from mtah.common import OFFICIAL, X
from mtah import dataset, model, noise
import possession_metrics as pm

sys.path.insert(0, os.path.join(OFFICIAL, "starter"))
from baseline import smooth  # noqa: E402  the official baseline's smoothing, reused as-is

TEST_SEED = 100
N_FOLDS = 3


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


def spell_f1(gt, pred):
    r = pm.evaluate(gt, pred)
    return r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"] - r["n_gt_spells"]


def fold_of(index_in_source, source):
    return index_in_source % N_FOLDS


def split():
    """[(test_games, train_games)] per fold, each game as (source, id, loader)."""
    games = dataset.games()
    position = {}
    tagged = []
    for g in games:
        position[g[0]] = position.get(g[0], -1) + 1
        tagged.append((fold_of(position[g[0]], g[0]), g))
    return [([g for f, g in tagged if f == k], [g for f, g in tagged if f != k])
            for k in range(N_FOLDS)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", action="store_true")
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()

    from mtah.run import PARAMS
    grid = [PARAMS]
    if args.grid:
        grid += [model.DecodeParams(s, t, f) for s, t, f in
                 itertools.product((5, 10, 20, 40), (5, 10, 20), (10, 20, 40))]

    scores = {i: {} for i in range(len(grid))}   # i -> source -> [(spell, macro, count_err)]
    base = {}
    for k, (test, train) in enumerate(split()):
        items = [dataset.item(s, g, load, seed) for s, g, load in train for seed in dataset.seeds(s)]
        clf = model.fit([(it["feats"], it["frames"]) for it in items], [it["labels"] for it in items])
        for source, game_id, load in test:
            it = dataset.item(source, game_id, load, TEST_SEED)
            prob = model.proba(clf, it["feats"])
            for i, params in enumerate(grid):
                pred = model.decode(prob, it["frames"], it["half_runs"], it["n_frames"], params)
                scores[i].setdefault(source, []).append(spell_f1(it["labels"], pred))
            if args.baseline:
                m = noise.degrade(load(), np.random.default_rng(TEST_SEED))
                base.setdefault(source, []).append(spell_f1(m.labels, baseline_codes(m)))
        print("fold {} done ({} test games)".format(k, len(test)), flush=True)

    def line(name, by_source):
        parts = ["{} spellF1 {:.4f} macroF1 {:.4f} count err {:+.0f}".format(
            src, *np.mean(v, axis=0)) for src, v in sorted(by_source.items())]
        print("  {:<58} {}".format(name, " | ".join(parts)))

    if base:
        line("baseline_smooth50", base)
    ranked = sorted(scores, key=lambda i: -np.mean([r[0] for v in scores[i].values() for r in v]))
    for i in ranked[:12]:
        line(str(grid[i]).replace("DecodeParams", ""), scores[i])


if __name__ == "__main__":
    main()
