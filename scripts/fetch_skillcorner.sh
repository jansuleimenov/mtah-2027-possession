#!/usr/bin/env bash
# SkillCorner open data at a pinned commit. Only data/ is checked out (the repo's notebook paths
# exceed Windows' path limit), and the LFS tracking files come from GitHub's media host because
# `git lfs pull` returns nothing for this repository.
set -euo pipefail
COMMIT=4340d274572876239c154c90bc507a9b3250a656
DEST=external/skillcorner

GIT_LFS_SKIP_SMUDGE=1 git clone -q --filter=blob:none --no-checkout \
  https://github.com/SkillCorner/opendata "$DEST"
git -C "$DEST" config core.longpaths true
git -C "$DEST" sparse-checkout set data/matches data/matches.json
GIT_LFS_SKIP_SMUDGE=1 git -C "$DEST" checkout -q "$COMMIT"

for dir in "$DEST"/data/matches/*/; do
  id=$(basename "$dir")
  rel="data/matches/$id/${id}_tracking_extrapolated.jsonl"
  curl -sfL -o "$DEST/$rel" "https://media.githubusercontent.com/media/SkillCorner/opendata/$COMMIT/$rel"
  echo "$id $(wc -c < "$DEST/$rel") bytes"
done
