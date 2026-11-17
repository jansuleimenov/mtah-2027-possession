# Solomon's Team: frame-level possession from automatic tracking

More than a Hack 2027, Phase 1. Team Solomon's Team.

## Approach

We treat possession as a segmentation problem, not a per-frame classification problem, because
`spellF1` scores whole possessions. The pipeline has three stages:

1. **Team-level features per frame**, computed from the automatic tracking (ATD) and the Smart
   Tagging. No player identities are used: the ATD has none, and its 46 track slots swap and
   fragment, so every feature is an aggregate over a team.
2. **A gradient-boosted classifier** (scikit-learn `HistGradientBoostingClassifier`) gives each
   frame probabilities for H, A and D.
3. **A three-state Viterbi decoder** turns those probabilities into a label sequence. It charges a
   fixed cost for every H↔A switch and for every change between alive and dead. This is what
   removes the frame-to-frame flicker that makes a per-frame model report several times too many
   possessions.

Why this rather than smoothing: a majority-vote window (the published baseline) has one knob and
trades flicker against late transitions everywhere at once. The decoder keeps a transition sharp
when the evidence for it is strong and suppresses it when the evidence is weak.

## Assets used and dropped

| asset | used | how |
|---|---|---|
| tactical ATD (players + ball) | yes | all features |
| Smart Tagging | yes | set-piece start frames as a restart signal; the opening kickoff of each half |
| tactical homography | no | the ATD is already in pitch coordinates |
| tactical and broadcast video | no | no GPU available to us; the ATD already encodes what we would extract |
| broadcast ATD | no | not synchronised to the tactical timeline |

## Features

All computed inside each half, never across halftime, with x flipped per half so the home team
always attacks towards +x (attack direction read from the home team's mean position in the 10 s
after the kickoff tag).

- **Ball**: visible flag, frames since last seen, last seen position, speed, out-of-bounds flag,
  distance to the nearest touchline or goal line.
- **Contest**: nearest home and away player distance to the ball and their difference; player
  counts per team within 2 m and 5 m of the ball.
- **Teams**: tracked player count, centroid x, spread, mean speed and forward velocity per team;
  share of players standing still (< 1 m/s).
- **Context**: centred rolling means over ±0.5 s, ±2 s and ±5 s of the nearest-team vote, ball
  visibility, overall speed, distance difference and ball speed.
- **Smart Tagging**: frames until the next SET PIECES tag and since the last one.

## Training data

None of it comes from the provided match.

| source | matches | labels | who labelled |
|---|---|---|---|
| [Metrica Sports sample data](https://github.com/metrica-sports/sample-data) | 3 | derived by our code from the matches' human event data | Metrica Sports (events) |
| [SkillCorner open data](https://github.com/SkillCorner/opendata), A-League 2024/25 | 20 | derived by our code from SkillCorner's phases-of-play files | SkillCorner (automatic, product data) |

- **Metrica**: possession goes to the team of each controlled event (pass, carry, shot, set piece,
  recovery, and ball lost, which means the team had it until then). Duels never change
  possession. The ball is dead from a ball-out, a foul, an offside or a shot out or in, until the
  next set piece or goal kick.
- **SkillCorner**: a phase of play exists only while the ball is in play and names the team in
  possession; between phases the label is D. Tracking is resampled from 10 to 25 fps.

### Making the external tracking look like the competition ATD

The external tracking is cleaner than the competition's, and in Metrica's data the ball is removed
during almost every stoppage, so a model trained on it would learn that "no ball" means "dead
ball". We degrade every training game before use:

- ball removed in bursts (mean 4 s) until 41% of graded frames have no ball, the rate DATA.md
  gives for the competition, with bursts three times likelier in crowds and twice as likely at
  high ball speed;
- on 60% of stoppages the ball is shown, held at its last position, as a placed ball would be;
- 7% of player observations removed in bursts, positions given a slow drift (sd 0.2 m);
- set-piece tags moved 3.3 s earlier (as the competition's kickoff tags are) and jittered
  (sd 1 s), 15% dropped, 3 false tags per half.

A check that this is realistic: the official baseline, run on our degraded external games, scores
spellF1 0.19–0.24, against its published 0.2433 on the competition match.

## Tuning

Every choice was made on a three-fold benchmark over the 23 external games, scored with the
organisers' `possession_metrics.py`. The provided match and its public five minutes were used
only to check that the output file is well formed.

| | Metrica spellF1 | SkillCorner spellF1 |
|---|---|---|
| official baseline (smooth50), after the noise fix below | 0.2390 | 0.1890 |
| our submission, after the noise fix | 0.5260 | 0.5961 |

Decoder costs: H↔A 20, alive→dead 10, dead→alive 40, chosen with the two sources weighted equally.

The public five minutes were scored once, after the file was final, as a correctness check:
spellF1 0.62 (12 possessions against the truth's 17), which confirms frames are aligned and the
team labels are not swapped. Nothing was chosen or changed on that number.

### A bug the competition data exposed

Our first noise model added independent per-frame jitter to every position. After differencing,
0.2 m of jitter reads as about 7 m/s, so every training player looked like a sprinter and the
competition match, with realistic speeds, looked like a standstill: the first file labelled 79%
of frames as dead ball. A diagnostic that compares feature medians on the competition match with
the training range (`scripts/diagnose.py`, aggregates only) found it. Position error is now smooth
drift (knots one second apart, linearly interpolated), and a regression test checks that standing
players stay standing.

## Possession rules implemented

- A possession changes only when the other team gains control; a loose ball stays with the last
  team (encoded in the training labels; the decoder's switch cost resists brief contacts).
- Before each half's opening kickoff (Smart Tagging "Kick Off (Start)"), every frame is D.
- Outside the two halves every frame is X.

## Limitations

- Our training labels come from two other definitions of possession (Metrica events, SkillCorner
  phases). The competition's ground truth comes from corrected tracking, a third definition. The
  places most likely to disagree are restarts and contested balls.
- The noise model is calibrated to published aggregates, not to the competition data itself.
- With more time we would add video-based ball detection to fill the 41% of frames without a ball.

## Reproduction

Code: https://github.com/jansuleimenov/mtah-2027-possession. Its README lists the steps: install
the pinned dependencies, fetch the two open datasets at pinned commits, then one command
regenerates the file and prints its sha256.
