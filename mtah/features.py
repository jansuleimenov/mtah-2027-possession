"""Per-frame features from team-level tracking. No player identities are used anywhere.

Everything is computed inside each half (never across halftime) and in an attack-normalised
frame: x is flipped per half so the home team always attacks towards +x, which keeps "who is
pushing forward" meaningful across the two halves.
"""

import numpy as np
import pandas as pd

from mtah.common import AWAY, FPS, HOME

NEAR = (2.0, 5.0)          # metres, player-count radii around the ball
WINDOWS = (12, 50, 125)    # centred context windows, frames (~0.5 s, 2 s, 5 s)
SEEN_CAP = 250             # frames-since-ball-seen is capped at 10 s


def build(match):
    """(features DataFrame indexed by frame, frame index array) for frames inside a half."""
    blocks = []
    for half, start, stop in match.half_runs():
        kickoff = match.meta.get("kickoff", {}).get(half, start)
        blocks.append(_half_features(match, start, stop, kickoff - start))
    frame = pd.concat(blocks)
    return frame, frame.index.to_numpy()


def _half_features(match, start, stop, kickoff=0):
    length, width = match.pitch
    ball = match.ball[start:stop].copy()
    players = match.players[start:stop].copy()
    home, away = match.team == HOME, match.team == AWAY

    # Attack direction: the home team's mean x over the first 10 s from the kickoff. Home on the
    # left (x < half pitch) attacks +x; otherwise flip so it does.
    early = np.nanmean(players[kickoff: kickoff + 10 * FPS][:, home, 0])
    if early > length / 2:
        ball[:, 0] = length - ball[:, 0]
        players[..., 0] = length - players[..., 0]

    out = {}
    seen = ~np.isnan(ball[:, 0])
    out["ball_seen"] = seen.astype(float)
    out["since_seen"] = _since(seen)
    held = pd.DataFrame(ball).ffill().bfill().to_numpy()
    out["ball_x"], out["ball_y"] = held[:, 0], held[:, 1]
    out["ball_out"] = ((held[:, 0] < -0.5) | (held[:, 0] > length + 0.5) |
                       (held[:, 1] < -0.5) | (held[:, 1] > width + 0.5)).astype(float)
    step = np.r_[0, np.linalg.norm(np.diff(held, axis=0), axis=1)] * FPS
    out["ball_speed"] = _roll(step, 5)

    dist = np.linalg.norm(players - held[:, None, :], axis=2)
    d_home = np.nanmin(np.where(home, dist, np.inf), axis=1)
    d_away = np.nanmin(np.where(away, dist, np.inf), axis=1)
    d_home[~np.isfinite(d_home)] = 60.0
    d_away[~np.isfinite(d_away)] = 60.0
    out["d_home"], out["d_away"] = d_home, d_away
    out["d_diff"] = d_away - d_home          # > 0 means home is closer
    for r in NEAR:
        out["n_home_%g" % r] = np.sum((dist < r) & home, axis=1).astype(float)
        out["n_away_%g" % r] = np.sum((dist < r) & away, axis=1).astype(float)

    speed = np.linalg.norm(np.diff(players, axis=0, prepend=players[:1]), axis=2) * FPS
    speed = np.clip(speed, 0, 12)
    for name, mask in (("home", home), ("away", away)):
        xs = players[:, mask, 0]
        out["n_%s" % name] = np.sum(~np.isnan(xs), axis=1).astype(float)
        out["cx_%s" % name] = np.nan_to_num(np.nanmean(xs, axis=1), nan=length / 2)
        out["sx_%s" % name] = np.nan_to_num(np.nanstd(xs, axis=1))
        out["v_%s" % name] = np.nan_to_num(np.nanmean(speed[:, mask], axis=1))
        out["vx_%s" % name] = _roll(np.gradient(out["cx_%s" % name]) * FPS, 25)
    out["v_all"] = (out["v_home"] + out["v_away"]) / 2
    out["v_max"] = np.nan_to_num(np.nanmax(speed, axis=1))
    out["cx_diff"] = out["cx_home"] - out["cx_away"]

    # Context: who has been nearer the ball, and how fast the game is, around each frame.
    nearer = np.sign(np.clip(out["d_diff"], -3, 3)) * seen
    for w in WINDOWS:
        out["near_vote_%d" % w] = _roll(nearer, 2 * w + 1)
        out["seen_frac_%d" % w] = _roll(seen.astype(float), 2 * w + 1)
        out["v_all_%d" % w] = _roll(out["v_all"], 2 * w + 1)
        out["d_diff_%d" % w] = _roll(np.clip(out["d_diff"], -15, 15), 2 * w + 1)
        out["ball_speed_%d" % w] = _roll(out["ball_speed"], 2 * w + 1)
    # Dead-ball cues: restarts happen at the boundary and with the players standing still.
    out["d_boundary"] = np.minimum.reduce([held[:, 0], length - held[:, 0],
                                           held[:, 1], width - held[:, 1]])
    out["slow_frac"] = np.nanmean(np.where(np.isnan(speed), np.nan, speed < 1.0), axis=1)
    out["slow_frac"] = _roll(np.nan_to_num(out["slow_frac"]), 25)

    # Smart Tagging SET PIECES: frames to the next restart and since the last one, in this half.
    tags = np.asarray(match.meta.get("setpieces", []), dtype=int)
    tags = tags[(tags >= start) & (tags < stop)] - start
    out["sp_to_next"], out["sp_since"] = _tag_distances(tags, stop - start)

    return pd.DataFrame(out, index=np.arange(start, stop))


def _tag_distances(tags, n):
    """(frames until next tag, frames since last tag), both capped at SEEN_CAP."""
    idx = np.arange(n)
    nxt = np.full(n, SEEN_CAP, dtype=float)
    prev = np.full(n, SEEN_CAP, dtype=float)
    if len(tags):
        pos = np.searchsorted(tags, idx)
        has_next = pos < len(tags)
        nxt[has_next] = tags[pos[has_next]] - idx[has_next]
        prev_pos = np.searchsorted(tags, idx, side="right") - 1
        has_prev = prev_pos >= 0
        prev[has_prev] = idx[has_prev] - tags[prev_pos[has_prev]]
    return np.minimum(nxt, SEEN_CAP), np.minimum(prev, SEEN_CAP)


def _roll(values, window):
    """Centred rolling mean, edges handled with shorter windows."""
    return pd.Series(values).rolling(window, center=True, min_periods=1).mean().to_numpy()


def _since(seen):
    """Frames since the ball was last seen, capped."""
    idx = np.where(seen, np.arange(len(seen)), -10 ** 9)
    last = np.maximum.accumulate(idx)
    return np.minimum(np.arange(len(seen)) - last, SEEN_CAP).astype(float)
