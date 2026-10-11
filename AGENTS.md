# AGENTS.md

Operating agreement for agents working in this repo. Read this first, then read `README.md`.

## Goal

Time series analysis + forecasting of air raid alerts in Ukraine, evaluated on the latest
data with zero data leakage. Ship a working, reproducible pipeline and a short write-up.

## Current layout

```
README.md        # problem, data sources, evaluation rules
AGENTS.md        # this file
opencode.json    # agent tooling config
data/raw/        # fetched data + manifest.json (gitignored)
data/processed/  # derived features (gitignored)
src/             # pipeline code
notebooks/       # exploration only; nothing final lives here
reports/         # figures + findings.md
tests/           # self-checks, runnable via `python -m pytest -q`
```

Missing directories are created when first needed — don't pre-scaffold.

## Commands

- Python: system `python3` (3.14). Use `uv` if installing packages; pin them in
  `pyproject.toml` when (and only when) a dependency is actually needed. Start with
  stdlib + pandas/numpy; add statsmodels/scikit-learn/lightgbm only if used.
- Tests: `python -m pytest -q`. One self-check per non-trivial module is enough.
- Data fetch scripts must be idempotent and safe to re-run: they skip or upsert, and
  always refresh `data/raw/manifest.json`.

## Hard rules (violating these fails the project)

1. **No leakage.** Nothing trained on or used as input may exist after the forecast origin.
   Scalers/encoders/imputers are fit on the training window only. Labels may use the
   future; features may not.
2. **Holdout is the latest period**, time-ordered, untouched until final evaluation.
   No random shuffling of time series. Use rolling-origin backtests for tuning.
3. **Never commit** `data/raw/`, `data/processed/`, tokens, or credentials. Tokens come
   from env vars (`ALERTS_IN_UA_TOKEN`, `UKRAINE_ALARM_TOKEN`). `.gitignore` enforces this.
4. **Data provenance:** every dataset you use gets a manifest entry (source, query,
   fetch time, row count, timestamp range). Claims in `reports/findings.md` cite it.
5. **Baselines first.** Seasonal-naive / hour-of-week profile before any fancy model,
   and always compared against in the report.
6. Determinism: fixed seeds, sorted joins, UTC timestamps, no wall-clock in features.

## Conventions

- Smallest working diff. No speculative abstractions, no config for values that never
  change, no frameworks. Delete dead code instead of preserving it "for later".
- Timestamps: `pandas` UTC-aware datetimes end to end.
- Prefer flat modules in `src/` (`fetch.py`, `features.py`, `evaluate.py`, ...) over
  packages with one class each.
- Comments explain *why*, never *what*. Mark deliberate ceilings with
  `ponytail: <ceiling>; upgrade path: <what to add>`.

## Definition of done

- `python -m pytest -q` passes.
- Reproducible run: `python -m src.fetch && python -m src.evaluate` produces the numbers
  in `reports/findings.md` (figures in `reports/figures/`).
- `reports/findings.md` states: data window and source, forecast setup and origins,
  metrics vs. baselines on the holdout, leakage checklist (explicit "what is known at
  forecast time" table), and honest limitations.
