"""Per-frame classifier plus a Viterbi decoder that turns probabilities into clean possessions.

spellF1 rewards whole possessions, not frames, so the decoder is where most of the score is: it
charges a fixed cost for every state change, which removes the flicker that sinks the baseline
(131 possessions in 5 minutes against a true 17). The costs are tuned on external games only.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from mtah.common import H, A, D, X

STATES = (H, A, D)
SUBSAMPLE = 5  # train on every 5th frame: neighbours are near-duplicates


@dataclass
class DecodeParams:
    switch: float = 40.0     # cost of H <-> A
    to_dead: float = 20.0    # cost of alive -> D
    from_dead: float = 20.0  # cost of D -> alive
    temperature: float = 1.0  # emission log-probs are divided by this


def fit(feature_frames, label_arrays, seed=0):
    """A classifier over H/A/D from a list of (features, frames) and full-length label arrays."""
    xs, ys = [], []
    for (feats, frames), labels in zip(feature_frames, label_arrays):
        y = labels[frames]
        keep = (y != X)
        keep[np.arange(len(keep)) % SUBSAMPLE != 0] = False
        xs.append(feats.to_numpy()[keep])
        ys.append(y[keep])
    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
        early_stopping=False, random_state=seed)
    model.fit(np.concatenate(xs), np.concatenate(ys))
    model.feature_names = list(feature_frames[0][0].columns)
    return model


def proba(model, feats):
    """(n, 3) probabilities in STATES order."""
    p = model.predict_proba(feats.to_numpy())
    order = [list(model.classes_).index(s) for s in STATES]
    return p[:, order]


def decode(prob, frames, half_runs, n_frames, params=DecodeParams()):
    """Full-length label codes: Viterbi inside each half, X outside."""
    out = np.full(n_frames, X, dtype=np.int8)
    index = np.full(n_frames, -1)
    index[frames] = np.arange(len(frames))
    cost = np.array([[0, params.switch, params.to_dead],
                     [params.switch, 0, params.to_dead],
                     [params.from_dead, params.from_dead, 0]], dtype=float)
    for _, start, stop in half_runs:
        rows = index[start:stop]
        emit = np.log(np.clip(prob[rows], 1e-6, 1)) / params.temperature
        out[start:stop] = np.array(STATES)[_viterbi(emit, cost)]
    return out


def _viterbi(emit, cost):
    """Best path maximising sum(emit) - sum(transition cost). emit (T, 3), cost (3, 3)."""
    t, k = emit.shape
    back = np.empty((t, k), dtype=np.int8)
    score = emit[0].copy()
    for i in range(1, t):
        cand = score[:, None] - cost          # from-state x to-state
        back[i] = np.argmax(cand, axis=0)
        score = cand[back[i], np.arange(k)] + emit[i]
    path = np.empty(t, dtype=np.int8)
    path[-1] = int(np.argmax(score))
    for i in range(t - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path
