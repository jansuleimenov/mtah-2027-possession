"""All external games, degraded and featurised, cached on disk.

Feature extraction is the slow step (~20 s a game), so each (game, noise seed) is computed once
and cached under a key that changes whenever features.py or noise.py changes.
"""

import hashlib
import os

import numpy as np
import pandas as pd

from mtah.common import EXTERNAL
from mtah import external, features, noise, skillcorner

CACHE = os.path.join(EXTERNAL, "cache", "features")
METRICA_SEEDS = (1, 2)   # Metrica: 3 games, so two degraded copies each
SKILLCORNER_SEEDS = (1,)


def games():
    """[(source, game_id, loader)] for every external game."""
    out = [("metrica", g, lambda g=g: external.load(g)) for g in external.GAMES]
    out += [("skillcorner", str(m), lambda m=m: skillcorner.load(m))
            for m in skillcorner.match_ids()]
    return out


def seeds(source):
    return METRICA_SEEDS if source == "metrica" else SKILLCORNER_SEEDS


def _version():
    here = os.path.dirname(os.path.abspath(__file__))
    digest = hashlib.sha256()
    for name in ("features.py", "noise.py", "external.py", "skillcorner.py"):
        digest.update(open(os.path.join(here, name), "rb").read())
    return digest.hexdigest()[:10]


def item(source, game_id, loader, seed):
    """dict(feats DataFrame, frames, labels, half_runs, n_frames) for one degraded game."""
    path = os.path.join(CACHE, "{}_{}_{}_{}.npz".format(source, game_id, seed, _version()))
    if os.path.isfile(path):
        try:
            z = np.load(path, allow_pickle=False)
            feats = pd.DataFrame(z["x"], columns=list(z["columns"]), index=z["frames"])
            return dict(feats=feats, frames=z["frames"], labels=z["labels"],
                        half_runs=[tuple(r) for r in z["half_runs"]], n_frames=int(z["n_frames"]))
        except (EOFError, ValueError, OSError, KeyError):
            pass  # truncated or half-written cache file: rebuild it below
    m = noise.degrade(loader(), np.random.default_rng(seed))
    feats, frames = features.build(m)
    feats = feats.astype(np.float32)  # same values whether read from cache or freshly built
    os.makedirs(CACHE, exist_ok=True)
    # Write then rename, so a second process never reads a half-written file.
    tmp = "{}.{}.tmp.npz".format(path[:-4], os.getpid())
    np.savez_compressed(tmp, x=feats.to_numpy(np.float32), columns=np.array(feats.columns, dtype=str),
                        frames=frames, labels=m.labels, half_runs=np.array(m.half_runs()),
                        n_frames=m.n_frames)
    os.replace(tmp, path)
    return dict(feats=feats, frames=frames, labels=m.labels, half_runs=m.half_runs(),
                n_frames=m.n_frames)
