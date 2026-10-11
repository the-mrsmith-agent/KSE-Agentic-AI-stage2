"""Model ladder for the alert forecasts: two baselines, then two learners.

Both learners read the same `FEATURES` matrix built by `src/features.build` (the daily
learners read its day-level subset, `DAY_FEATURES`), so the report can compare *learners*
rather than two feature pipelines. Everything here is a pure function or an sklearn
factory: no classes, no state outside the fitted sklearn estimator, nothing fitted on
anything but the training window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, PoissonRegressor

from src.features import FEATURES

# Daily targets have one row per origin x region, so the hourly features carry no
# day-level information: `hour_local` is constant-noise across the collapse,
# `minutes_to_sunset`/`is_night` describe one specific hour of the 24. They must not enter
# a daily model (they would fit the arbitrary "first hour" of the day). `day_of_week`,
# `is_weekend`, `month`, `is_holiday`, `day_length_min` are properties of the origin's
# local calendar day, not of an hour, so they stay.
HOUR_FEATURES = ["hour_local", "minutes_to_sunset", "is_night"]
DAY_FEATURES = [f for f in FEATURES if f not in HOUR_FEATURES]

_PROFILE_KEYS = ["region", "day_of_week", "hour_local"]


def daily_panel(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse a `build()` frame to one row per origin x region.

    Day features are taken verbatim from the origin's first (00:00 local) hour row, not
    aggregated, so the panel describes the calendar day the forecast is made on. Labels are
    summed over the 24 forecast hours. No row is fabricated: the collapsed index is exactly
    the origins x regions present in `frame`.
    """
    first = frame.sort_values(["origin", "region", "hour"]).groupby(["origin", "region"], as_index=False).head(1)
    sums = frame.groupby(["origin", "region"], as_index=False).agg(
        y_starts_day=("y_start", "sum"), y_minutes_day=("y_minutes", "sum")
    )
    out = first[["origin", "region", *DAY_FEATURES]].merge(sums, on=["origin", "region"], how="inner")
    return out.sort_values(["origin", "region"]).reset_index(drop=True)


def hour_of_week_profile(train_frame: pd.DataFrame) -> pd.DataFrame:
    """Baseline A: empirical P(alert starts) per (region, local weekday, local hour).

    Fit on training rows only. The two fallback levels are precomputed here as broadcast
    columns, so prediction is one lookup and one map — no recompute that could see eval data.
    """
    cells = train_frame.groupby(_PROFILE_KEYS, as_index=False)["y_start"].mean().rename(columns={"y_start": "p"})
    cells["p_region"] = cells["region"].map(train_frame.groupby("region")["y_start"].mean())
    cells["p_global"] = float(train_frame["y_start"].mean())
    return cells


def predict_hour_of_week(profile: pd.DataFrame, frame: pd.DataFrame) -> np.ndarray:
    """Baseline A prediction: cell mean -> region mean -> global mean. Length == len(frame).

    The two fallback levels live as broadcast columns of `profile`, so they are read with a
    separate map instead of the cell merge (a left join would NaN them out on an unseen cell).
    """
    cells = profile[_PROFILE_KEYS + ["p"]]
    region_mean = profile.drop_duplicates("region").set_index("region")["p_region"]
    global_mean = float(profile["p_global"].iloc[0]) if len(profile) else 0.0
    merged = frame[_PROFILE_KEYS].merge(cells, on=_PROFILE_KEYS, how="left")
    filled = merged["p"].fillna(merged["region"].map(region_mean)).fillna(global_mean)
    return filled.to_numpy(np.float64)


def seasonal_naive(train_frame: pd.DataFrame, frame: pd.DataFrame) -> np.ndarray:
    """Baseline B: the `y_start` observed at hour - 7 days for the same region; 0.0 if the
    training window does not cover that timestamp (quiet regime / start of the series)."""
    lookup = train_frame.groupby(["region", "hour"])["y_start"].mean()
    key = pd.MultiIndex.from_frame(frame[["region", "hour"]].assign(hour=frame["hour"] - pd.Timedelta("7D")))
    return lookup.reindex(key).fillna(0.0).to_numpy(np.float64)


def duration_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows where the duration label is defined: a start happened and was neither censored
    (`naive` end) nor implausibly long (>24 h), so excluding them avoids training on imputed
    ends."""
    return frame[(frame["y_start"] == 1) & frame["y_dur_log"].notna()]


def duration_naive(train_frame: pd.DataFrame, frame: pd.DataFrame) -> np.ndarray:
    """Duration baseline: training-window median log-duration-at-start, broadcast.

    ponytail: one constant per region-pooled series, no timing signal; upgrade path: make the
    median conditional on hour/region once the empirical quantiles show real spread.
    """
    median = duration_rows(train_frame)["y_dur_log"].median()
    return np.full(len(frame), 0.0 if pd.isna(median) else float(median))


def make_poisson(**kw) -> PoissonRegressor:
    """Model 1 (counts/minutes): log-link Poisson, coefficients expoundable as log-rate
    effects. Target must be >= 0 (y_starts_day, y_minutes_day). `alpha` is the L2 penalty
    that keeps the unscaled lag magnitudes in check. `newton-cholesky` because the raw
    (unscaled) lag features make lbfgs crawl (thousands of iterations); the exact Hessian
    converges in ~10 and is deterministic, so no seed is needed.
    ponytail: newton-cholesky factorizes a p x p matrix, so it wants features < rows;
    upgrade path: switch to lbfgs + a train-fitted scaler if the feature count ever grows.
    """
    params = dict(alpha=1.0, solver="newton-cholesky", max_iter=1000)
    params.update(kw)
    return PoissonRegressor(**params)


def make_logistic(**kw) -> LogisticRegression:
    """Model 1 (hourly start probability). Features are used raw: no scaler, so coefficients
    stay log-odds per original unit (a scaler fitted on train would be fine leakage-wise but
    would cost the interpretability that is the point of Model 1). `newton-cholesky` for the
    same reason as the Poisson fit. Seed fixed for the rare stochastic solvers.
    """
    params = dict(C=1.0, solver="newton-cholesky", max_iter=1000, random_state=0)
    params.update(kw)
    return LogisticRegression(**params)


def make_histgb(loss: str = "poisson", quantile: float | None = None, **kw) -> HistGradientBoostingRegressor:
    """Model 2: gradient boosting on the same matrix. `loss="poisson"` for counts/minutes,
    `loss="quantile"` with q in {0.1, 0.5, 0.9} for log-duration. Handles NaN natively, so
    the imputation in `fit_predict` is only needed by the linear models. `early_stopping` is
    off: an internal random validation split would make the fit depend on row order.
    """
    params = dict(
        loss=loss,
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=0,
    )
    if loss == "quantile":
        if quantile not in (0.1, 0.5, 0.9):
            raise ValueError(f"quantile must be 0.1/0.5/0.9, got {quantile!r}")
        params["quantile"] = quantile  # target range: log1p(minutes), ~[0, log(1440)]
    params.update(kw)
    return HistGradientBoostingRegressor(**params)


def fit_predict(model, train_frame: pd.DataFrame, frame: pd.DataFrame, target: str) -> np.ndarray:
    """Fit `model` on train rows, predict `frame` on the same feature matrix.

    Hourly frames carry all of `FEATURES`; daily panels carry `DAY_FEATURES`. Selecting the
    columns present keeps one code path for both. NaN is possible in `hours_since_last_alert`
    for the first days of the series; the linear models reject it, so it is filled with the
    train median (an imputer fit on the training window only — never on the eval frame).
    """
    cols = [c for c in FEATURES if c in train_frame.columns]
    assert len(cols) in (len(FEATURES), len(DAY_FEATURES)), "unexpected feature columns"
    X = train_frame[cols].to_numpy(np.float64)
    median = np.nanmedian(X, axis=0)
    median = np.where(np.isnan(median), 0.0, median)  # all-NaN column in train: 0 is as good as any
    model.fit(np.where(np.isnan(X), median, X), train_frame[target].to_numpy())
    X_eval = frame[cols].to_numpy(np.float64)
    return model.predict(np.where(np.isnan(X_eval), median, X_eval))
