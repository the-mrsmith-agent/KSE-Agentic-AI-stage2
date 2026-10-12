"""One command for the whole pipeline: fetch -> eda -> evaluate -> brief.

    python -m src.run_all

Each step is the same command you would type by hand, run as a subprocess (no shared module
caches, no fork-after-import surprises) and stopped at the first failure with its own output
intact. Order is the dependency order: `brief` reads `reports/predictions.csv` that
`evaluate` writes.

`src.fetch` hits the network and refreshes the upstream snapshot, so re-running can shift
every number in the reports; run the individual modules if you want to keep the data as-is.
"""

from __future__ import annotations

import subprocess
import sys

STEPS = ("src.fetch", "src.eda", "src.evaluate", "src.brief")


def main() -> None:
    for step in STEPS:
        print(f"\n=== {step} ===", flush=True)
        subprocess.run([sys.executable, "-m", step], check=True)


if __name__ == "__main__":
    main()
