"""Self-checks for src/models.py. Offline synthetic frames only: no data/raw, no network."""

import pathlib
import sys
import zoneinfo

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.features import FEATURES, daily_origin
from src.models import (
    DAY_FEATURES,
    daily_panel,
    duration_naive,
    duration_rows,
    fit_predict,
    hour_of_week_profile,
    make_histgb,
    make_logistic,
    make_poisson,
    predict_hour_of_week,
    seasonal_naive,
)

KYIV = zoneinfo.ZoneInfo("Europe/Kyiv")
ORIGIN = daily_origin("2024-06-10")  # Monday


def toy_frame(regions=("A",), hours=24, start=ORIGIN, seed=0):
    """A build()-shaped frame: 24h per region, every FEATURES column, zero labels."""
    idx = pd.MultiIndex.from_product(
        [list(regions), pd.date_range(start, periods=hours, freq="h")], names=["region", "hour"]
    )
    df = pd.DataFrame(index=idx).reset_index()
    df.insert(0, "origin", start)
    rng = np.random.default_rng(seed)
    for col in FEATURES:
        df[col] = rng.normal(size=len(df))
    local = df["hour"].dt.tz_convert(KYIV)
    df["hour_local"] = local.dt.hour
    df["day_of_week"] = local.dt.dayofweek
    df["is_weekend"] = (local.dt.dayofweek >= 5).astype(int)
    df["month"] = local.dt.month
    df["is_holiday"] = 0
    df["day_length_min"] = df.groupby("region").cumcount()
    df["y_start"] = 0
    df["y_minutes"] = 0.0
    df["y_dur_log"] = np.nan
    return df[["origin", "region", "hour", *FEATURES, "y_start", "y_minutes", "y_dur_log"]]


def test_day_features_drop_hour_only_columns():
    assert set(DAY_FEATURES) == set(FEATURES) - {"hour_local", "minutes_to_sunset", "is_night"}
    for col in ("day_of_week", "is_weekend", "month", "is_holiday", "day_length_min"):
        assert col in DAY_FEATURES
    assert [c for c in FEATURES if c in DAY_FEATURES] == DAY_FEATURES  # FEATURES order kept


def test_daily_panel_sums_hours_and_keeps_first_row_day_features():
    frame = toy_frame(regions=("A", "B"), hours=24)
    frame.loc[frame["hour"] == ORIGIN, "y_start"] = 2  # the panel must sum, not average
    frame["y_minutes"] = 10.0
    panel = daily_panel(frame)

    assert list(panel.columns) == ["origin", "region", *DAY_FEATURES, "y_starts_day", "y_minutes_day"]
    assert len(panel) == 2  # one row per region for this origin
    assert panel["y_starts_day"].tolist() == [2, 2]
    assert panel["y_minutes_day"].tolist() == [240.0, 240.0]
    # day features come from the 00:00 local row, not from an aggregate over the day
    assert panel["day_length_min"].tolist() == [0, 0]
    assert panel["day_of_week"].tolist() == [0, 0]


def test_daily_panel_row_count_is_origins_times_regions():
    frame = pd.concat([toy_frame(regions=("A", "B", "C"), start=ORIGIN + pd.Timedelta(days=d)) for d in range(3)])
    panel = daily_panel(frame)
    assert len(panel) == 3 * 3
    assert panel[["origin", "region"]].duplicated().sum() == 0
    assert panel.equals(panel.sort_values(["origin", "region"]).reset_index(drop=True))  # deterministic


def test_hour_of_week_profile_recovers_known_cell_mean():
    train = toy_frame(regions=("A",), hours=48)  # Monday + Tuesday, local hours 0..23 twice
    train.loc[train["hour_local"] == 3, "y_start"] = 1
    profile = hour_of_week_profile(train)

    assert len(profile) == 48  # 2 weekdays x 24 local hours, one region
    assert profile.loc[profile["hour_local"] == 3, "p"].eq(1.0).all()
    assert profile.loc[profile["hour_local"] == 9, "p"].eq(0.0).all()
    assert profile["p_region"].dropna().unique().tolist() == [2 / 48]
    assert profile["p_global"].dropna().unique().tolist() == [2 / 48]


def test_predict_hour_of_week_falls_back_and_never_nans():
    train = toy_frame(regions=("A", "B"), hours=24)  # Monday only, so Tuesday cells are all unseen
    train.loc[(train["region"] == "A") & (train["hour_local"] == 3), "y_start"] = 1
    train.loc[(train["region"] == "B") & (train["hour_local"] == 5), "y_start"] = 1
    train.loc[(train["region"] == "B") & (train["hour_local"] == 6), "y_start"] = 1
    profile = hour_of_week_profile(train)

    monday = toy_frame(regions=("A",), hours=24)
    assert predict_hour_of_week(profile, monday)[monday["hour_local"] == 3].tolist() == [1.0]  # known cell

    tuesday = toy_frame(regions=("A", "B", "Z"), hours=24, start=ORIGIN + pd.Timedelta("1D"))
    pred = predict_hour_of_week(profile, tuesday)

    assert len(pred) == len(tuesday) and not np.isnan(pred).any()
    assert pred[tuesday["region"] == "A"].tolist() == [1 / 24] * 24  # -> region mean
    assert pred[tuesday["region"] == "B"].tolist() == [2 / 24] * 24  # -> region mean
    assert pred[tuesday["region"] == "Z"].tolist() == [3 / 48] * 24  # unseen region -> global mean


def test_seasonal_naive_reads_seven_days_earlier():
    train = toy_frame(regions=("A",), hours=24, start=ORIGIN - pd.Timedelta("7D"))
    train["y_start"] = train["hour_local"]  # distinct known value per row
    frame = toy_frame(regions=("A",), hours=24)

    pred = seasonal_naive(train, frame)
    assert pred.tolist() == frame["hour_local"].tolist()


def test_seasonal_naive_is_zero_when_timestamp_absent():
    train = toy_frame(regions=("A",), hours=24, start=ORIGIN - pd.Timedelta("7D"))
    train["y_start"] = 1
    frame = toy_frame(regions=("A", "Z"), hours=24, start=ORIGIN + pd.Timedelta("1D"))
    assert seasonal_naive(train, frame).tolist() == [0.0] * len(frame)


def test_duration_rows_keep_only_uncensored_starts():
    frame = toy_frame(regions=("A",), hours=4)
    frame["y_start"] = [1, 0, 1, 1]
    frame["y_dur_log"] = [0.5, np.nan, np.nan, 2.0]  # row 2 = censored start, row 1 = no start
    kept = duration_rows(frame)
    assert kept["y_dur_log"].tolist() == [0.5, 2.0]


def test_duration_naive_broadcasts_train_median():
    train = toy_frame(regions=("A",), hours=5)
    train["y_start"] = [1, 1, 1, 0, 0]
    train["y_dur_log"] = [1.0, 2.0, 3.0, 9.0, 9.0]  # the 9s are not duration rows
    frame = toy_frame(regions=("A",), hours=4)
    assert duration_naive(train, frame).tolist() == [2.0] * 4
    empty = train.assign(y_start=0)
    assert duration_naive(empty, frame).tolist() == [0.0] * 4  # no median -> 0, never NaN


def test_fit_predict_poisson_on_daily_panel():
    frame = pd.concat([toy_frame(regions=("A", "B"), start=ORIGIN + pd.Timedelta(days=d), seed=d) for d in range(4)])
    frame.loc[frame["hour"].dt.hour % 6 == 0, "y_start"] = 1
    frame["y_minutes"] = 15.0 * frame["y_start"]
    panel = daily_panel(frame)
    train, test = panel.iloc[:4], panel.iloc[4:]

    pred = fit_predict(make_poisson(), train, test, "y_starts_day")
    assert len(pred) == len(test) and not np.isnan(pred).any()
    assert (pred >= 0).all()  # Poisson mean is non-negative by construction


def test_fit_predict_logistic_and_nan_tolerance():
    train = toy_frame(regions=("A", "B"), hours=48)
    train["y_start"] = (train["hour_local"] % 5 == 0).astype(int)
    train.loc[train["region"] == "A", "hours_since_last_alert"] = np.nan  # first days of the series
    test = toy_frame(regions=("A", "B"), hours=24, seed=7)
    test["hours_since_last_alert"] = np.nan

    pred = fit_predict(make_logistic(), train, test, "y_start")
    assert len(pred) == len(test)
    assert np.isfinite(pred).all() and ((pred >= 0) & (pred <= 1)).all()
    # `predict` would return class labels; the caller needs the rate, so the result must not
    # collapse to a two-valued 0/1 vector.
    assert len(np.unique(pred)) > 2


def test_histgb_fits_and_predicts_without_nan():
    train = toy_frame(regions=("A", "B"), hours=48, seed=3)
    train["y_start"] = 1
    train["y_dur_log"] = np.random.default_rng(4).uniform(0.5, 4.0, len(train))
    rows = duration_rows(train)
    train.loc[train["region"] == "A", "cloud_mean_24h"] = np.nan  # HistGB takes NaN natively

    median = fit_predict(make_histgb(loss="quantile", quantile=0.5), rows, rows, "y_dur_log")
    assert len(median) == len(rows) and not np.isnan(median).any()

    counts = fit_predict(make_histgb(), daily_panel(train), daily_panel(train), "y_starts_day")
    assert len(counts) == 2 and not np.isnan(counts).any()


def test_histgb_rejects_unlisted_quantile():
    with pytest.raises(ValueError, match="quantile"):
        make_histgb(loss="quantile", quantile=0.25)
