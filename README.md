# More than a Hack 2027: frame-level ball possession

Entry for [More than a Hack 2027](https://github.com/metrica-sports/more-than-a-hack-2027), Phase 1:
label every frame of FC Barcelona vs Athletic Club as `H`, `A`, `D` or `X`.

## Approach in one paragraph

Team-level features from the automatic tracking (no player identities): ball state, distances
from each team to the ball, player counts near the ball, team speeds and shape, plus centred
context windows. A gradient-boosted classifier turns them into per-frame H/A/D probabilities,
and a 3-state Viterbi decoder with switching costs turns those into whole possessions, which is
what `spellF1` scores. The model is trained only on other matches: Metrica Sports' three open
sample games, with possession labels derived from their event data and the tracking degraded to
match the competition ATD (41% missing ball, dropped player tracks, no "ball gone = dead" shortcut).
Every hyperparameter is chosen on a leave-one-game-out benchmark over those games.

## Reproduce

Python 3.12.

```bash
git clone --recurse-submodules <this repo> && cd <this repo>
python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt   # Windows
git clone https://github.com/metrica-sports/sample-data external/metrica-sample-data
git -C external/metrica-sample-data checkout e706dd506b360d69d9d123d5b8026e7294b13996
bash scripts/fetch_skillcorner.sh          # 20 matches, ~1.8 GB, pinned commit

# the competition bundle, unpacked so that <data>/tactical/BAR-ATH_tactical_atd.txt exists
python -m mtah.run --data <data> --out out/<team>.csv
python vendor/official/evaluation/validate_submission.py out/<team>.csv
```

`mtah.run` prints the sha256 of the file it wrote; it must equal the submitted file's.

## Layout

| path | what |
|---|---|
| `mtah/external.py` | Metrica open games: loaders and event-to-label derivation |
| `mtah/noise.py` | degrade clean tracking to look like the competition ATD |
| `mtah/features.py` | per-frame team-level features |
| `mtah/model.py` | classifier and Viterbi decoder |
| `mtah/bench.py` | leave-one-game-out benchmark on the external games (all tuning happens here) |
| `mtah/competition.py` | load the competition ATD and Smart Tagging |
| `mtah/run.py` | train on external games, label the competition match, write the CSV |
| `vendor/official` | the organisers' repository (scorer, frame alignment, loaders), as a submodule |

The competition data is not in this repository and must not be: it is licensed for the
competition only.
