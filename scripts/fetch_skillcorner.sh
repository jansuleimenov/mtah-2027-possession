#!/usr/bin/env bash
# SkillCorner open data at a pinned commit. Only data/ is checked out (the repo's notebook paths
# exceed Windows' path limit), and the LFS tracking files come from GitHub's media host because
# `git lfs pull` returns nothing for this repository.
# The checkout leaves a small LFS pointer (oid sha256 + size) at each tracking path. Each file is
# downloaded only while it is still a pointer, retried on a dropped connection, checked against the
# pointer's sha256, and only then moved into place, so the script is safe to re-run after a failure.
set -euo pipefail
COMMIT=4340d274572876239c154c90bc507a9b3250a656
DEST=external/skillcorner

if [ ! -d "$DEST/.git" ]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone -q --filter=blob:none --no-checkout \
    https://github.com/SkillCorner/opendata "$DEST"
  git -C "$DEST" config core.longpaths true
  git -C "$DEST" sparse-checkout set data/matches data/matches.json
  GIT_LFS_SKIP_SMUDGE=1 git -C "$DEST" checkout -q "$COMMIT"
fi

for dir in "$DEST"/data/matches/*/; do
  id=$(basename "$dir")
  rel="data/matches/$id/${id}_tracking_extrapolated.jsonl"
  file="$DEST/$rel"
  # A failed earlier run may have left no file at all: put the pointer back.
  [ -s "$file" ] || GIT_LFS_SKIP_SMUDGE=1 git -C "$DEST" checkout -q "$COMMIT" -- "$rel"
  if head -c 40 "$file" | grep -q '^version https://git-lfs'; then
    oid=$(sed -n 's/^oid sha256://p' "$file")
    curl -sfL --retry 5 --retry-all-errors --retry-delay 5 -o "$file.part" \
      "https://media.githubusercontent.com/media/SkillCorner/opendata/$COMMIT/$rel"
    got=$(sha256sum "$file.part" | cut -d' ' -f1)
    if [ "$got" != "$oid" ]; then
      rm -f "$file.part"
      echo "$id: sha256 $got does not match the pointer's $oid" >&2
      exit 1
    fi
    mv "$file.part" "$file"
  fi
  echo "$id $(wc -c < "$file") bytes"
done
