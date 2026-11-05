"""External training and tuning data: Metrica Sports' open sample games.

github.com/metrica-sports/sample-data -- three anonymised professional matches with synchronised
tracking (players and ball, 25 fps) and human-annotated event data. They are *other games*, so
the challenge rules allow training and tuning on them, including on labels derived from their
human event data.

Frame-level H/A/D labels are derived from the events with the challenge's own definition
(CHALLENGE.md): possession changes only when the other team gains *controlled* possession; a
loose or contested ball stays with the last team that had it; a dead ball is D.
"""

import json
import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from mtah.common import EXTERNAL, AWAY, HOME, H, A, D, X, Match

SAMPLE = os.path.join(EXTERNAL, "metrica-sample-data", "data")
PITCH = (105.0, 68.0)
GAMES = ("game1", "game2", "game3")

# An event of these types means its team had the ball under control at the event's start frame.
# BALL LOST is included on purpose: the team had the ball until it lost it, and by the challenge
# definition it keeps possession until the opponent actually controls it (a RECOVERY or any later
# controlled event). CHALLENGE is excluded: a duel is not control.
CONTROL_TYPES = {"PASS", "CARRY", "SHOT", "SET PIECE", "RECOVERY", "BALL LOST"}


# -- loading -----------------------------------------------------------------------------------

CACHE = os.path.join(EXTERNAL, "cache")


def load(game):
    """A labelled Match for one of GAMES, cached as .npz after the first parse."""
    path = os.path.join(CACHE, game + ".npz")
    if os.path.isfile(path):
        z = np.load(path)
        return Match("metrica-" + game, z["ball"], z["players"], z["team"], z["period"], PITCH,
                     z["labels"])
    match = _parse(game)
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(path, ball=match.ball, players=match.players, team=match.team,
                        period=match.period, labels=match.labels)
    return match


def _parse(game):
    if game in ("game1", "game2"):
        match, events = _load_csv_game(int(game[-1]))
    elif game == "game3":
        match, events = _load_epts_game()
    else:
        raise KeyError("unknown game {!r}; have {}".format(game, GAMES))
    match.labels, match.period = derive_labels(events, match.n_frames)
    match.meta["events"] = events
    return match


def _load_csv_game(number):
    folder = os.path.join(SAMPLE, "Sample_Game_{}".format(number))
    stem = os.path.join(folder, "Sample_Game_{}_".format(number))

    sides = []
    for side, code in (("Home", HOME), ("Away", AWAY)):
        frame = pd.read_csv(stem + "RawTrackingData_{}_Team.csv".format(side), skiprows=2)
        cols = list(frame.columns)
        names = [c for c in cols if c.startswith("Player")]
        xy = np.stack([frame[[n, cols[cols.index(n) + 1]]].to_numpy(float) for n in names],
                      axis=1)
        ball = frame[["Ball", cols[cols.index("Ball") + 1]]].to_numpy(float)
        sides.append((xy, np.full(len(names), code, dtype=np.int8), ball))

    players = np.concatenate([sides[0][0], sides[1][0]], axis=1)
    team = np.concatenate([sides[0][1], sides[1][1]])
    ball = sides[0][2]
    match = Match("metrica-game{}".format(number), _metres(ball), _metres(players), team,
                  np.zeros(len(ball), dtype=np.int8), PITCH)

    raw = pd.read_csv(stem + "RawEventsData.csv")
    events = pd.DataFrame({
        "team": np.where(raw["Team"] == "Home", HOME, AWAY),
        "type": raw["Type"].str.upper(),
        "subtypes": [frozenset(str(s).upper().split("-")) if isinstance(s, str) else frozenset()
                     for s in raw["Subtype"]],
        "start": raw["Start Frame"].astype(int) - 1,
        "end": np.maximum(raw["End Frame"].astype(int), raw["Start Frame"].astype(int)) - 1,
        "period": raw["Period"].astype(int),
    })
    return match, events


def _load_epts_game():
    folder = os.path.join(SAMPLE, "Sample_Game_3")
    root = ET.parse(os.path.join(folder, "Sample_Game_3_metadata.xml")).getroot()

    team_ids = [t.get("id") for t in root.iter("Team")]
    player_team = {p.get("id"): (HOME if p.get("teamId") == team_ids[0] else AWAY)
                   for p in root.iter("Player")}
    channel_player = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}
    slots = sorted(player_team)
    slot_of = {pid: i for i, pid in enumerate(slots)}

    # Each format spec covers a frame range with its own ordered player list (substitutions).
    specs = []
    for spec in root.iter("DataFormatSpecification"):
        order = [channel_player[ref.get("playerChannelId")]
                 for ref in spec.iter("PlayerChannelRef")
                 if ref.get("playerChannelId").endswith("_x")]
        specs.append((int(spec.get("startFrame")), int(spec.get("endFrame")),
                      [slot_of[p] for p in order]))

    lines = open(os.path.join(folder, "Sample_Game_3_tracking.txt")).read().splitlines()
    n = len(lines)
    players = np.full((n, len(slots), 2), np.nan)
    ball = np.full((n, 2), np.nan)
    for line in lines:
        frame_id, groups, ball_xy = line.split(":")
        frame_id = int(frame_id)
        order = next(o for s, e, o in specs if s <= frame_id <= e)
        i = frame_id - 1
        for slot, group in zip(order, groups.split(";")):
            x, y = group.split(",")[:2]
            players[i, slot] = (float(x), float(y))
        bx, by = ball_xy.split(",")[:2]
        ball[i] = (float(bx), float(by))

    team = np.array([player_team[p] for p in slots], dtype=np.int8)
    match = Match("metrica-game3", _metres(ball), _metres(players), team,
                  np.zeros(n, dtype=np.int8), PITCH)

    data = json.load(open(os.path.join(folder, "Sample_Game_3_events.json")))["data"]
    events = pd.DataFrame({
        "team": [HOME if e["team"]["id"] == team_ids[0] else AWAY for e in data],
        "type": [e["type"]["name"].upper() for e in data],
        "subtypes": [_subtypes(e.get("subtypes")) for e in data],
        "start": [int(e["start"]["frame"]) - 1 for e in data],
        "end": [max(int(e["end"]["frame"] or e["start"]["frame"]), int(e["start"]["frame"])) - 1
                for e in data],
        "period": [int(e["period"]) for e in data],
    })
    return match, events


def _subtypes(value):
    if not value:
        return frozenset()
    if isinstance(value, dict):
        value = [value]
    return frozenset(s["name"].upper() for s in value)


def _metres(xy):
    """Normalised 0-1 coordinates to metres, any leading shape."""
    return xy * np.array(PITCH)


# -- labels ------------------------------------------------------------------------------------

def is_dead_trigger(kind, subtypes):
    """True if this event puts the ball out of play, and at which end ('start' or 'end')."""
    if kind == "BALL OUT":
        return "end"
    if kind == "SHOT" and (subtypes & {"OUT", "GOAL"}):
        return "end"
    if kind == "FAULT RECEIVED":
        return "start"
    if kind == "BALL LOST" and "OFFSIDE" in subtypes:
        return "end"
    return None


def derive_labels(events, n_frames):
    """(labels, period) arrays from event data.

    Each half runs from its first event (the kickoff) to its last event's end frame; everything
    outside is X / period 0, i.e. ungraded, the same as the competition ground truth.
    """
    labels = np.full(n_frames, X, dtype=np.int8)
    period = np.zeros(n_frames, dtype=np.int8)

    for half in sorted(events["period"].unique()):
        evs = events[events["period"] == half].sort_values("start", kind="stable")
        first, last = int(evs["start"].min()), int(evs["end"].max()) + 1
        last = min(last, n_frames)

        points = []  # (frame, order, state) -- order keeps a dead trigger after a same-frame touch
        for row in evs.itertuples(index=False):
            if row.type in CONTROL_TYPES:
                points.append((row.start, 0, H if row.team == HOME else A))
            where = is_dead_trigger(row.type, row.subtypes)
            if where:
                points.append((row.end if where == "end" else row.start, 1, D))
        points.sort()

        for (frame, _, state), nxt in zip(points, points[1:] + [(last, 0, None)]):
            lo, hi = max(frame, first), min(nxt[0], last)
            if hi > lo:
                labels[lo:hi] = state
        period[first:last] = half
    return labels, period
