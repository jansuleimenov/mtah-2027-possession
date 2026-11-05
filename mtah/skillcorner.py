"""Second external source: SkillCorner open data (10 A-League 2024/25 matches, MIT licence).

github.com/SkillCorner/opendata -- broadcast tracking at 10 fps plus a phases-of-play file.
Labels: a phase of play exists only while the ball is in play and names the team in possession,
so inside a phase the label is that team and between phases it is D. Tracking is resampled to
25 fps so features and decoder costs mean the same thing as on the competition data. Restarts
(phase starts after a stoppage) stand in for Smart Tagging's SET PIECES.
"""

import json
import os

import numpy as np
import pandas as pd

from mtah.common import EXTERNAL, AWAY, FPS, HOME, H, A, D, X, Match

ROOT = os.path.join(EXTERNAL, "skillcorner", "data")
CACHE = os.path.join(EXTERNAL, "cache")
SRC_FPS = 10
RESTART_GAP = 1.0  # seconds of no phase before a phase start counts as a restart


def match_ids():
    with open(os.path.join(ROOT, "matches.json")) as f:
        return [m["id"] for m in json.load(f)]


def load(match_id):
    path = os.path.join(CACHE, "sc_{}.npz".format(match_id))
    if os.path.isfile(path):
        z = np.load(path)
        m = Match("skillcorner-{}".format(match_id), z["ball"], z["players"], z["team"],
                  z["period"], tuple(z["pitch"]), z["labels"])
        m.meta["setpieces"] = z["setpieces"]
        return m
    m = _parse(match_id)
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(path, ball=m.ball, players=m.players, team=m.team, period=m.period,
                        pitch=np.array(m.pitch), labels=m.labels, setpieces=m.meta["setpieces"])
    return m


def _parse(match_id):
    folder = os.path.join(ROOT, "matches", str(match_id))
    stem = os.path.join(folder, str(match_id))
    info = json.load(open(stem + "_match.json"))
    length, width = float(info["pitch_length"]), float(info["pitch_width"])
    home_id = info["home_team"]["id"]
    slots = [p["id"] for p in info["players"]]
    slot_of = {pid: i for i, pid in enumerate(slots)}
    team = np.array([HOME if p["team_id"] == home_id else AWAY for p in info["players"]],
                    dtype=np.int8)

    rows = [json.loads(line) for line in open(stem + "_tracking_extrapolated.jsonl")]
    n_src = max(r["frame"] for r in rows) + 1
    src_ball = np.full((n_src, 2), np.nan)
    src_players = np.full((n_src, len(slots), 2), np.nan)
    for r in rows:
        b = r.get("ball_data") or {}
        if b.get("x") is not None:
            src_ball[r["frame"]] = (b["x"] + length / 2, b["y"] + width / 2)
        for p in r.get("player_data") or []:
            if p.get("player_id") in slot_of and p.get("x") is not None:
                src_players[r["frame"], slot_of[p["player_id"]]] = (p["x"] + length / 2,
                                                                   p["y"] + width / 2)

    src_labels = np.full(n_src, X, dtype=np.int8)
    src_period = np.zeros(n_src, dtype=np.int8)
    for per in info["match_periods"]:
        lo, hi = per["start_frame"], min(per["end_frame"] + 1, n_src)
        src_labels[lo:hi] = D
        src_period[lo:hi] = per["period"]
    phases = pd.read_csv(stem + "_phases_of_play.csv")
    restarts = []
    last_end = {}
    for ph in phases.sort_values("frame_start").itertuples(index=False):
        code = H if ph.team_in_possession_id == home_id else A
        src_labels[ph.frame_start:ph.frame_end + 1] = code
        prev = last_end.get(ph.period)
        if prev is None or (ph.frame_start - prev) / SRC_FPS >= RESTART_GAP:
            restarts.append(ph.frame_start)
        last_end[ph.period] = ph.frame_end
    src_labels[src_period == 0] = X

    # Resample 10 -> 25 fps: positions linearly interpolated, labels from the nearest frame.
    t = np.arange(int((n_src - 1) * FPS / SRC_FPS) + 1) * SRC_FPS / FPS
    lo = np.floor(t).astype(int)
    hi = np.minimum(lo + 1, n_src - 1)
    w = (t - lo)[:, None]
    ball = src_ball[lo] * (1 - w) + src_ball[hi] * w
    players = src_players[lo] * (1 - w[..., None]) + src_players[hi] * w[..., None]
    nearest = np.minimum(np.round(t).astype(int), n_src - 1)
    labels = src_labels[nearest]
    period = src_period[nearest]
    setpieces = np.round(np.array(restarts) * FPS / SRC_FPS).astype(int)

    m = Match("skillcorner-{}".format(match_id), ball, players, team, period, (length, width),
              labels)
    m.meta["setpieces"] = np.sort(setpieces)
    return m
