"""Degrade clean external tracking until it looks like the competition's automatic ATD.

The external games are corrected tracking: the ball is visible on ~90% of live frames and removed
on almost every dead frame. The competition ATD is uncorrected: the ball is missing on 41% of
graded frames, disproportionately in crowds and in the air, and nothing says it disappears when
play stops. A model trained on the clean data would learn "no ball = dead ball", which is false
on the target. This module removes that shortcut and adds the target's kind of noise.

Every figure below is a property of the competition data published in DATA.md (41% missing ball,
~20.5 of 22 players tracked), not something measured on the competition data by us.
"""

from dataclasses import dataclass, replace

import numpy as np

from mtah.common import D


@dataclass
class NoiseParams:
    ball_missing_target: float = 0.41   # DATA.md: ball absent on 41% of graded frames
    ball_burst_mean: float = 30.0       # mean dropout length, frames
    crowd_radius: float = 3.0           # metres; crowded = >= crowd_n players this close
    crowd_n: int = 3
    crowd_boost: float = 3.0            # dropout hazard multiplier in a crowd
    fast_ball: float = 15.0             # m/s; hazard multiplier above this speed (balls in air)
    fast_boost: float = 2.0
    dead_ball_shown: float = 0.6        # share of dead stretches where the ball stays visible
    ball_jitter: float = 0.3            # metres, sd
    player_missing: float = 0.07        # DATA.md: mean 20.5 of 22 tracked
    player_burst_mean: float = 50.0
    player_jitter: float = 0.2
    # Simulated Smart Tagging set pieces: automatic, so imperfect. Tags are second-accurate
    # (DATA.md), miss some restarts and add a few spurious ones.
    tag_jitter: float = 25.0            # frames, sd
    tag_miss: float = 0.15
    tag_false_per_half: float = 3.0


def degrade(match, rng, params=NoiseParams()):
    """A copy of `match` with the ball, players and set-piece tags degraded. Labels untouched."""
    meta = dict(match.meta)
    if "setpieces" in meta:
        meta["setpieces"] = _degrade_tags(meta["setpieces"], match, rng, params)
    ball = _show_dead_ball(match.ball.copy(), match.labels, rng, params)
    ball = _drop_ball(ball, match.players, match.period > 0, rng, params)
    ball += rng.normal(0, params.ball_jitter, ball.shape)

    players = match.players.copy()
    t, s, _ = players.shape
    gone = _bursts((t, s), params.player_missing, params.player_burst_mean, rng)
    players[gone] = np.nan
    players += rng.normal(0, params.player_jitter, players.shape)
    return replace(match, ball=ball, players=players, meta=meta)


def _degrade_tags(frames, match, rng, params):
    keep = frames[rng.random(len(frames)) >= params.tag_miss]
    keep = keep + np.round(rng.normal(0, params.tag_jitter, len(keep))).astype(int)
    false = [rng.integers(start, stop, rng.poisson(params.tag_false_per_half))
             for _, start, stop in match.half_runs()]
    return np.sort(np.clip(np.concatenate([keep] + false), 0, match.n_frames - 1))


def _show_dead_ball(ball, labels, rng, params):
    """On a share of dead stretches, hold the ball at its last seen position (a placed ball)."""
    if labels is None:
        return ball
    dead = labels == D
    edges = np.diff(np.concatenate(([0], dead.astype(np.int8), [0])))
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if rng.random() > params.dead_ball_shown:
            continue
        seen = np.flatnonzero(~np.isnan(ball[:start, 0]))
        if len(seen):
            ball[start:stop] = ball[seen[-1]]
    return ball


def _drop_ball(ball, players, graded, rng, params):
    """Burst dropouts, likelier in crowds and at speed, scaled to hit the target missing rate."""
    t = len(ball)
    dist = np.linalg.norm(players - ball[:, None, :], axis=2)
    crowded = np.nansum(dist < params.crowd_radius, axis=1) >= params.crowd_n
    speed = np.r_[0, np.linalg.norm(np.diff(ball, axis=0), axis=1) * 25]
    weight = np.ones(t)
    weight[crowded] *= params.crowd_boost
    weight[np.nan_to_num(speed) > params.fast_ball] *= params.fast_boost

    already = np.isnan(ball[:, 0])
    base = already[graded].mean()
    need = max(params.ball_missing_target - base, 0.0) / max(1 - base, 1e-9)
    # Start a burst at frame i with hazard weight_i * k; bursts of mean length L cover ~ k*w*L.
    k = need / (params.ball_burst_mean * weight[graded].mean() * (1 - need) + 1e-9)
    starts = np.flatnonzero(rng.random(t) < weight * k)
    lengths = rng.geometric(1.0 / params.ball_burst_mean, len(starts))
    mask = _paint(starts, lengths, t)
    ball[mask] = np.nan
    return ball


def _bursts(shape, rate, mean_len, rng):
    """Boolean mask over (T, S): per-column bursts covering about `rate` of frames."""
    t, s = shape
    out = np.zeros(shape, dtype=bool)
    k = rate / (mean_len * (1 - rate))
    for j in range(s):
        starts = np.flatnonzero(rng.random(t) < k)
        out[:, j] = _paint(starts, rng.geometric(1.0 / mean_len, len(starts)), t)
    return out


def _paint(starts, lengths, n):
    """Boolean mask with [start, start+length) set for each burst."""
    delta = np.zeros(n + 1, dtype=np.int32)
    np.add.at(delta, starts, 1)
    np.add.at(delta, np.minimum(starts + lengths, n), -1)
    return np.cumsum(delta[:-1]) > 0
