# Air-raid alerts in Ukraine — time series analysis & forecasting

Forecasting Ukraine's air-raid sirens from the public volunteer alert history (24 oblasts,
since Feb 2022): **when alerts are likely to start, how many to expect, and how long the next
one will last** — one forecast per region per day, evaluated on the most recent data with no
peeking at the future.

## Read this first

| | |
|---|---|
| **Plain-language results** (no statistics needed) | [`reports/brief.md`](reports/brief.md) |
| **Full technical write-up** (data window, setup, metrics, leakage checklist, limitations) | [`reports/findings.md`](reports/findings.md) |
| **How the models evolved** (readable model first, then the iteration) | [`reports/models.md`](reports/models.md) |
| **Exploration** — charts and numbers | [`reports/eda.md`](reports/eda.md), [`reports/figures/`](reports/figures/) |
| **Problem statement and data decisions** | [`GOAL.md`](GOAL.md) |
| **How this repo is built and worked on** | [`AGENTS.md`](AGENTS.md) |

## Run it

```sh
uv run python -m src.run_all    # fetch -> eda -> evaluate -> brief, the whole pipeline
uv run pytest -q                # self-checks
```

Step by step:

```sh
uv run python -m src.fetch      # alert + weather snapshot, writes data/raw/manifest.json
uv run python -m src.eda        # reports/eda.md and the exploration figures
uv run python -m src.evaluate   # reports/metrics.json, reports/predictions.csv, eval figures
uv run python -m src.brief      # reports/brief.md — the plain-language briefing
```

`src.fetch` is the only step that touches the network; everything else runs offline on
`data/raw/`. Re-running `src.fetch` refreshes the data and can shift every number in the
reports, which is why `findings.md` quotes the exact snapshot it was written against.

## What we found, in short

- **How many alerts to expect** — the models genuinely help: wrong by ~1.5 alerts per
  region-day against 2.2 for the simple "assume a typical recent day" guess (~11 vs 22
  countrywide).
- **Which hour an alert starts** — about half right: the hour is somewhat predictable, the
  minute is not.
- **How long an alert lasts** — essentially unsolved, barely better than answering "the usual
  length".
- Volume has grown ~2.3× since 2022; front-line oblasts spend ~4–6 hours a day under sirens.

Numbers and caveats: [`reports/brief.md`](reports/brief.md) ·
[`reports/findings.md`](reports/findings.md).

## Layout

```
src/fetch.py      alert snapshot + Open-Meteo download, idempotent, writes manifest.json
src/features.py   cleaning + the origin-guarded feature/target builder (no-leakage core)
src/eda.py        exploration figures + reports/eda.md
src/models.py     baselines and models (one shared feature matrix)
src/evaluate.py   time-ordered splits, rolling-origin backtest, metrics, figures
src/brief.py      plain-language briefing
src/run_all.py    runs the steps above in order
tests/            offline self-checks
```

## Data

Primary source: `volunteer_data_en.csv` from
[`Vadimkin/ukrainian-air-raid-sirens-dataset`](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset)
(no token needed); weather and daylight from Open-Meteo. Raw data lands in `data/raw/` and is
**never committed** — every fetch records source, query, fetch time, row counts and timestamp
range in `data/raw/manifest.json`. Provenance and cleaning rules: `reports/findings.md` §2.

## Limitations

Volunteer-collected data (some alerts missing), oblast granularity, weather enters as
past-24-hour lags rather than forecasts, Luhansk and Crimea absent. Full list with numbers:
`reports/findings.md` §6.
