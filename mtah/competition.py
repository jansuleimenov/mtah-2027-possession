"""Load the competition match (BAR-ATH, tactical camera) into a Match.

Only automatic inputs are read: the ATD and the Smart Tagging. Frame alignment and graded halves
come from the official games.py; the slot -> team table from the official baseline.
"""

import os
import sys

import numpy as np

from mtah.common import OFFICIAL, Match

sys.path.insert(0, os.path.join(OFFICIAL, "starter"))
import games  # noqa: E402
from baseline import team_of_each_track  # noqa: E402
import load_smart_tagging  # noqa: E402


def load(data_dir=None):
    meta_path, raw_path = games.require(["atd_meta", "atd_raw"], games.GAME, data_dir)
    length, width = games.GAMES[games.GAME]["pitch"]
    n = games.source_frames("tactical", games.GAME)
    team = team_of_each_track(meta_path)

    players = np.full((n, len(team), 2), np.nan)
    ball = np.full((n, 2), np.nan)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            i = int(parts[0]) - 1  # ATD frame_id counts from 1, video frames from 0
            groups = parts[1].split(";")
            values = np.array(",".join(groups).split(","), dtype=float)
            values = values.reshape(len(groups), -1)[:, :2]
            players[i, : len(groups)] = values
            ball[i] = np.array(parts[2].split(",")[:2], dtype=float)

    period = np.zeros(n, dtype=np.int8)
    for half, first, last in games.half_video_blocks(games.GAME):
        period[first:last + 1] = half

    match = Match(games.GAME, ball * [length, width], players * [length, width], team, period,
                  (length, width))
    match.meta["kickoff"], match.meta["setpieces"] = set_pieces(data_dir)
    return match


def set_pieces(data_dir=None):
    """({half: opening kickoff frame}, sorted start frames of every SET PIECES tag), both from
    the automatic Smart Tagging."""
    kickoff, starts = {}, []
    for tag in load_smart_tagging.load(games.GAME, data_dir):
        if tag["code"].lower() != "set pieces":
            continue
        starts.append(tag["start_frame"])
        tags = {k.lower(): (v or "").lower() for k, v in tag["tags"].items()}
        if tags.get("type") == "kick off (start)":
            half = 1 if tags.get("half", "").startswith("1") else 2
            kickoff.setdefault(half, tag["start_frame"])
    return kickoff, np.sort(np.array(starts, dtype=int))
