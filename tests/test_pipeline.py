"""Tests that run without the competition data.

The competition loader is exercised on a synthetic bundle written in the documented format
(DATA.md, starter/baseline.py), so the parser is checked without anyone opening the real files.
"""

import os

import numpy as np
import pytest

from mtah.common import A, D, H, X, Match
from mtah import model, noise, external


def _write_bundle(root, n_slots=4):
    folder = os.path.join(root, "tactical")
    os.makedirs(folder)
    with open(os.path.join(folder, "BAR-ATH_tactical_atd.xml"), "w") as f:
        f.write('<Root><Teams><Team id="T1"><Name>Home</Name></Team>'
                '<Team id="T2"><Name>Away</Name></Team></Teams><Players>')
        for s in range(n_slots):
            f.write('<Player id="P{}" teamId="{}"/>'.format(s, "T1" if s % 2 == 0 else "T2"))
        f.write("</Players></Root>")
    with open(os.path.join(folder, "BAR-ATH_tactical_atd.txt"), "w") as f:
        for frame_id in (11186, 11187):  # video frames 11185, 11186
            groups = ";".join("0.{0}1,0.{0}2,1.5".format(s + 1) for s in range(n_slots - 1))
            f.write("{}:{};NaN,NaN,NaN:0.5,0.25,0.0\n".format(frame_id, groups))
    with open(os.path.join(folder, "BAR-ATH_tactical_smart_tagging.xml"), "w") as f:
        f.write("<file><ALL_INSTANCES>"
                "<instance><code>SET PIECES</code><start>447.4</start><end>450</end>"
                "<label><group>Type</group><text>Kick Off (Start)</text></label>"
                "<label><group>Half</group><text>1st Half</text></label></instance>"
                "<instance><code>SET PIECES</code><start>4549.68</start><end>4552</end>"
                "<label><group>Type</group><text>Kick Off (Start)</text></label>"
                "<label><group>Half</group><text>2nd Half</text></label></instance>"
                "</ALL_INSTANCES></file>")


def test_competition_loader_on_synthetic_bundle(tmp_path):
    _write_bundle(str(tmp_path))
    from mtah import competition
    m = competition.load(str(tmp_path))
    assert m.n_frames == 199319
    assert list(m.team) == [0, 1, 0, 1]
    assert np.allclose(m.ball[11185], [0.5 * 105.3, 0.25 * 68.0])
    assert np.allclose(m.players[11185, 0], [0.11 * 105.3, 0.12 * 68.0])
    assert np.isnan(m.players[11185, 3]).all()           # NaN group
    assert np.isnan(m.ball[11184]).all()                 # no line for that frame
    assert m.period[11185] == 1 and m.period[11184] == 0
    assert m.period[110808] == 2 and m.period[190197] == 2 and m.period[190198] == 0
    assert m.meta["kickoff"] == {1: 11185, 2: 113742}
    assert list(m.meta["setpieces"]) == [11185, 113742]


def test_viterbi_removes_flicker():
    t = 400
    prob = np.full((t, 3), 0.05)
    truth = np.r_[np.zeros(200, int), np.ones(200, int)]
    prob[np.arange(t), truth] = 0.9
    flips = np.arange(5, 195, 10)
    prob[flips] = [0.05, 0.9, 0.05]                     # one-frame flickers to A
    codes = model.decode(prob, np.arange(t), [(1, 0, t)], t, model.DecodeParams(switch=10))
    assert (codes[:200] == H).all() and (codes[200:] == A).all()


def test_noise_hits_missing_ball_target():
    rng = np.random.default_rng(0)
    t, s = 20000, 22
    players = rng.uniform(0, 60, (t, s, 2))
    ball = rng.uniform(0, 60, (t, 2))
    labels = np.full(t, H, dtype=np.int8)
    labels[::50] = D
    m = Match("synthetic", ball, players, np.r_[np.zeros(11), np.ones(11)].astype(np.int8),
              np.ones(t, dtype=np.int8), labels=labels)
    out = noise.degrade(m, np.random.default_rng(1))
    missing = np.isnan(out.ball[:, 0]).mean()
    assert 0.33 < missing < 0.49


def test_labels_follow_challenge_definition():
    import pandas as pd
    ev = pd.DataFrame([
        dict(team=0, type="SET PIECE", subtypes=frozenset({"KICK OFF"}), start=10, end=10, period=1),
        dict(team=0, type="PASS", subtypes=frozenset(), start=10, end=20, period=1),
        dict(team=1, type="CHALLENGE", subtypes=frozenset({"AERIAL", "WON"}), start=30, end=30, period=1),
        dict(team=0, type="BALL LOST", subtypes=frozenset(), start=40, end=45, period=1),
        dict(team=1, type="RECOVERY", subtypes=frozenset(), start=60, end=60, period=1),
        dict(team=1, type="BALL OUT", subtypes=frozenset(), start=70, end=75, period=1),
        dict(team=0, type="SET PIECE", subtypes=frozenset({"THROW IN"}), start=90, end=90, period=1),
        dict(team=0, type="PASS", subtypes=frozenset(), start=90, end=100, period=1),
    ])
    labels, period = external.derive_labels(ev, 120)
    assert (labels[:10] == X).all()
    assert (labels[10:60] == H).all()    # duel and loose ball do not change possession
    assert (labels[60:75] == A).all()    # controlled recovery does
    assert (labels[75:90] == D).all()    # ball out until the restart
    assert (labels[90:101] == H).all()
    assert (labels[101:] == X).all() and period[100] == 1 and period[101] == 0
