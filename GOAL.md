# Air Raid Alerts Ukraine — Time Series Analysis

Mini pet-project (KSE Agentic AI, stage 2): time series analysis and forecasting of air
raid alerts in Ukraine — a real-world defense problem.

## Problem

Air raid sirens in Ukraine follow strong temporal structure: daily/weekly seasonality,
clustering during mass attacks, quiet stretches between them. The task is to turn the
alert history into useful forecasts: **when alerts are likely to start, how many to expect,
and how long the next alert will last**.

Concretely, build a pipeline that:

1. **Ingests** the most recent alert history available (per oblast and/or nationwide).
2. **Explores** it: seasonality, trends, clustering, gaps, changes in alert regimes.
3. **Forecasts** the latest period with at least one interpretable baseline
   (e.g. historical hour-of-week profile) and at least one real model
   (e.g. gradient boosting on lag features, or a count/event-time model).
4. **Evaluates** honestly on the most recent data (see below) and writes up findings.

## Data

**Primary source decided: `datasets/volunteer_data_en.csv`** from the public snapshot
`Vadimkin/ukrainian-air-raid-sirens-dataset` (auto-updated daily by CI). It is oblast-level
(23 oblasts + Kyiv City), consistently granular from 2022-02-25 to within hours of run time,
and needs no token. Weather/daylight come from Open-Meteo (free, no key).

Rejected for this build, kept as upgrade paths:

- **`alerts.in.ua` / `api.ukrainealarm.com` APIs** — no token in the environment.
  `ponytail: snapshot source; upgrade path: alerts.in.ua client when a token exists.`
- **`official_data_en.csv`** from the same repo — richest data, but it breaks from oblast to
  raion/hromada granularity in Dec 2025, i.e. exactly inside the holdout window. Training on
  oblast rows and evaluating on reconstructed raion rows would be a regime artifact, not signal.

Observed properties that drive cleaning (measured on the 2022-02-25 → 2026-10-11 pull):

- 25 units incl. 3 stray Luhansk rows (Luhansk/Crimea have permanent sirens and are absent) → drop.
- 5% of rows are `naive=True` (end message never arrived; end imputed as start + 30 min).
  Exactly 50% of alerts are ≤30 min, so these are **right-censored durations** — keep the
  start event, exclude the duration from duration labels.
- No overlapping consecutive alerts per region, so no de-duplication/merging pass needed.
- Volumes roughly doubled 2022 → 2026 (missile → mass-drone shift): the series is
  non-stationary and the holdout lives in a different regime than the early data.

Rules:

- Raw data lands in `data/raw/` and is **never committed**. Record the exact
  `data/raw/manifest.json`: source, query, fetch time, row counts, min/max timestamps.
- Gamekeeping: `fetch` is the only network entry point, idempotent, and always refreshes the
  manifest (upstream git SHA, fetch time, row counts, timestamp range).
- Prefer per-oblast alerts at alert-start/stop granularity. Downstream, aggregate freely
  (e.g. hourly alert-seconds per oblast).

Extensions included (all timestamp-safe under the cutoff below): Open-Meteo **weather lags**
(temp, cloud, wind, precip) and day length / minutes-to-sunset (known years ahead, and drone
waves are night-bound), plus a fixed Ukrainian holiday calendar.
Excluded: Telegram/DSNS posts and missile/drone event reports — no clean public feed.
`ponytail: weather as past-24h lags; upgrade path: Open-Meteo forecast API read at origin.`

## Forecast setup

- **Origin = 00:00 Europe/Kyiv, once per day; horizon = next 24 h; unit = region × hour.**
  One coherent setup for all three questions rather than three ad-hoc ones.
- Every evaluation row carries an explicit `origin`; the feature builder takes `origin` and
  asserts nothing it reads has a timestamp ≥ origin. That assert *is* the leakage test, and
  it is the only path features can be built through (train and eval share it).
- Targets:
  1. **counts** — alert starts per region per day (+ alert-minutes/day as a secondary target);
  2. **start timing** — hourly binary "an alert starts in this hour";
  3. **duration** — log-duration of the next alert, modelled *conditional on a start*
     (features known at start time), evaluated with MAE + split-conformal coverage.
- Feature families (all knowable at origin): calendar (dow/hour/month/season/weekend/UA
  holidays) · lags (starts in past 1/6/24/72 h, alert-minutes past 24 h, time since last
  alert) · count of regions alerting nationally in the past 3 h (mass attacks are nationwide,
  a strong leading signal) · trailing 28-day rate per region · weather lags · daylight.
- Stack: **pandas / numpy / scikit-learn / matplotlib** only.

## Model ladder

Baselines first, then two models that share one feature matrix (so the report compares
features vs learner, not two data pipelines):

- **Baseline A** — hour-of-week empirical profile per region, fit on the train window only.
- **Baseline B** — seasonal naive (same hour last week).
- **Model 1 (interpretable)** — `PoissonRegressor` for daily counts
  (log-link, coefficients expoundable) + `LogisticRegression` for hourly start probability.
- **Model 2 (real)** — `HistGradientBoostingRegressor`: `loss="poisson"` for counts/minutes,
  `loss="quantile"` (0.1/0.5/0.9) for log-duration.
- Duration intervals via **split conformal**: fit on train, calibrate residuals on the tune
  window, wrap the median prediction — ~10 lines, honest coverage numbers.

## Evaluation & no leakage

**Leakage is a hard failure, not a warning.** Everything a forecast uses must be knowable
at the moment the forecast is made.

- Choose an explicit **forecast origin** for every evaluation row. Any feature, label,
  scaler, encoder, or hyperparameter input with a timestamp after the origin is a leak.
- Fit preprocessing (scalers, target encoders, imputers) on the training window only.
- **Holdout = the latest period**, never shuffled and never trained on. Suggested split:
  train on everything up to `T-28 days`, tune on `T-28d..T-14d`, test on the last 14 days.
  If the data ends earlier than today, "latest period" means the latest period *in the data*.
- Report baselines alongside the model. If the model can't beat "alerts happen at the usual
  times", say so.
- Suggested metrics (pick per task, state why):
  - alert count per day/oblast: MAE / RMSE vs. seasonal naive;
  - alert start events: precision/recall/F1 within a tolerance window (±15–30 min);
  - duration: MAE on log-duration + empirical coverage;
  - calibration: predicted vs. observed alert-time share per oblast.
- Backtesting: rolling-origin evaluation (≥ 3 folds), never random K-fold on time.
  Folds are used for model/hyperparameter choice only; the last 14 days are touched once.

## Build plan

Flat modules (per `AGENTS.md`), built in this order:

1. `src/fetch.py` — snapshot + Open-Meteo download, idempotent, writes `manifest.json`.
2. `src/features.py` — clean (drop stray Luhansk rows, flag `naive` rows as censored, drop
   alerts >24 h from duration labels) and the origin-guarded feature/target builder.
3. `src/eda.py` — seasonality, clustering/gaps, regime shift 2022→2026, per-oblast burden.
4. `src/models.py` — baselines + the two models.
5. `src/evaluate.py` — splits, rolling-origin backtest, metrics, conformal calibration,
   figures to `reports/figures/`, numbers for `findings.md`.
6. `reports/findings.md` — data window + manifest citations, forecast setup and origins,
   metrics vs baselines on the holdout, an explicit **"known at forecast time"** table
   (each feature: timestamp rule, why safe), and honest limitations (volunteer-sourced data,
   censored ends, weather-as-lags, oblast granularity hides the post-Dec-2025 raion regime,
   Luhansk/Crimea absent).

Remaining ceilings, marked in code: snapshot instead of API; weather lags instead of
forecast-at-origin; no Telegram / missile-event signals.

## Status

Planned and agreed — implementation starts at `src/fetch.py`. Definition of done and
working agreement live in `AGENTS.md`.
