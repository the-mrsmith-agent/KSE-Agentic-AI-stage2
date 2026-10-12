"""Self-check for src.run_all: the step order matches the data dependencies.

Cheap and offline — it asserts the order instead of spawning the pipeline (fetch would hit
the network and the tests must not).
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.run_all import STEPS


def test_steps_are_ordered_by_dependency():
    assert STEPS[0] == "src.fetch"  # refreshes what everything else reads
    assert STEPS[-1] == "src.brief"  # reads reports/predictions.csv + metrics.json
    assert STEPS.index("src.evaluate") < STEPS.index("src.brief")
    assert len(set(STEPS)) == len(STEPS)  # no step runs twice
