"""Produce the submission file for BAR-ATH.

    python -m mtah.run --data C:/mtah-data --out out/<team>.csv

Trains on the external Metrica games only (never on anything from the competition match), then
labels the competition match. Prints summary lines only -- counts and validation -- never frame
values, so the output is safe to share.
"""

import argparse
import hashlib
import os

import numpy as np

from mtah.common import D, LABELS
from mtah import external, features, model, noise

TRAIN_SEEDS = (1, 2)
# Chosen on the external leave-one-game-out benchmark (mtah/bench.py --grid). Never tuned on the
# competition match or its public ground truth.
PARAMS = model.DecodeParams()


def train():
    feats, labels = [], []
    for game in external.GAMES:
        clean = external.load(game)
        for seed in TRAIN_SEEDS:
            m = noise.degrade(clean, np.random.default_rng(seed))
            feats.append(features.build(m))
            labels.append(m.labels)
    return model.fit(feats, labels)


def label(match, clf, params=PARAMS):
    feats, frames = features.build(match)
    codes = model.decode(model.proba(clf, feats), frames, match.half_runs(), match.n_frames, params)
    # Before each half's kickoff (from the automatic Smart Tagging) the ball is not in play.
    for half, start, stop in match.half_runs():
        kickoff = match.meta.get("kickoff", {}).get(half)
        if kickoff is not None and start < kickoff < stop:
            codes[start:kickoff] = D
    return codes


def write(path, codes):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, LABELS[c]) for i, c in enumerate(codes))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="folder holding tactical/ and broadcast/")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.environ["MTAH_DATA_DIR"] = args.data

    from mtah import competition  # after MTAH_DATA_DIR is set
    import possession_metrics as pm

    clf = train()
    match = competition.load(args.data)
    codes = label(match, clf)
    write(args.out, codes)

    runs = [(s, e) for _, s, e in match.half_runs()]
    stats = pm.possession_stats(codes, runs)
    graded = codes[np.concatenate([np.arange(s, e) for s, e in runs])]
    print("wrote {} rows -> {}".format(len(codes), args.out))
    print("graded frames: H {} A {} D {} X {}".format(
        *[int(np.sum(graded == k)) for k in range(4)]))
    print("possessions {}  mean {:.1f} s  home share {:.3f}".format(
        stats["n_spells"], stats["mean_duration_s"], stats["home_share"]))
    print("sha256 {}".format(hashlib.sha256(open(args.out, "rb").read()).hexdigest()))
    print("next: python vendor/official/evaluation/validate_submission.py {}".format(args.out))


if __name__ == "__main__":
    main()
