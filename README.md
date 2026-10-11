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

Use the most recent data you can get at run time. Candidates (in order of preference):

- **`alerts.in.ua` API** — historical alerts per oblast (token via `ALERTS_IN_UA_TOKEN`).
- **`api.ukrainealarm.com`** — Ukraine Alarm API (token via `UKRAINE_ALARM_TOKEN`).
- **Public snapshots** (fallback / reproducibility): `vadimkin/ukraine-air-raid-sirens-dataset`
  (GitHub), Kaggle / Hugging Face mirrors of alert histories.

Rules:

- API tokens live in env vars only — never in code, data, or git.
- Raw data lands in `data/raw/` and is **never committed**. Record the exact
  `data/raw/manifest.json`: source, query, fetch time, row counts, min/max timestamps.
- Prefer per-oblast alerts at alert-start/stop granularity. Downstream, aggregate freely
  (e.g. hourly alert-seconds per oblast).

Optional extensions (nice-to-have, not required): text/media signals that existed at
forecast time — official channel posts (Telegram `@air_alert_ua` / DSNS), weather,
missile/drone event reports. These must obey the same timestamp cutoff as everything else.

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

## Status

Scaffolding only — the pipeline, models, and write-up are to be built in this repo.
See `AGENTS.md` for the working agreement and current layout.
