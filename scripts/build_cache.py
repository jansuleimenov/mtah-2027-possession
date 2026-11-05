"""Build the feature cache for every external game, training seeds and the benchmark test seed."""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mtah import dataset  # noqa: E402
from mtah.bench import TEST_SEED  # noqa: E402

for source, game_id, load in dataset.games():
    for seed in tuple(dataset.seeds(source)) + (TEST_SEED,):
        t = time.time()
        dataset.item(source, game_id, load, seed)
        print("{} {} seed {} {:.0f}s".format(source, game_id, seed, time.time() - t), flush=True)
