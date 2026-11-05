"""Train the submission model twice and confirm the labels it produces are byte-identical.

The rules require the repository to regenerate the submitted file exactly. This checks the
training half of that on an external game, so it needs no competition data.
"""

import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mtah import dataset, model, run  # noqa: E402

source, game_id, load = dataset.games()[2]          # metrica game3
probe = dataset.item(source, game_id, load, 100)
digests = []
for attempt in range(2):
    clf = run.train()
    codes = model.decode(model.proba(clf, probe["feats"]), probe["frames"], probe["half_runs"],
                         probe["n_frames"], run.PARAMS)
    digests.append(hashlib.sha256(codes.tobytes()).hexdigest())
    print("run {}: {}".format(attempt + 1, digests[-1]), flush=True)
print("IDENTICAL" if digests[0] == digests[1] else "DIFFERENT")
sys.exit(0 if digests[0] == digests[1] else 1)
