"""Self-checks for src/evaluate.py. Offline synthetic frames only: no data/raw, no network."""

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.evaluate import (
    conformal_width,
    coverage,
    match_events,
    prf,
    rolling_folds,
    seasonal_naive_daily,
    split_days,
    train_window,
    usable_origins,
)


def alerts_frame(start="2025-01-01", end_day="2026-06-29", regions=("A", "B")):
    """Daily synthetic alerts, two regions, with a known data end at 2026-06-30 04:00 UTC."""
    days = pd.date_range(start, end_day, freq="D", tz="UTC")
    rows = [
        (region, d + pd.Timedelta("10h"), d + pd.Timedelta("10h30min"), False)
        for region in regions
        for d in days
    ]
    df = pd.DataFrame(rows, columns=["region", "started_at", "finished_at", "naive"])
    df.loc[len(df)] = ("A", pd.Timestamp("2026-06-30 03:30", tz="UTC"),
                       pd.Timestamp("2026-06-30 04:00", tz="UTC"), False)
    return df


def test_usable_origins_drops_horizon_past_data_end():
    a = alerts_frame()
    u = usable_origins(a)
    last_fin = a["finished_at"].max()
    assert all(o + pd.Timedelta("24h") <= last_fin for o in u)
    # the very next daily origin would run past the data end and must be absent
    nxt = u[-1] + pd.Timedelta("24h")
    assert nxt not in u
    assert nxt + pd.Timedelta("24h") > last_fin
    assert u == sorted(u)


def test_split_windows_are_disjoint_and_time_ordered():
    a = alerts_frame()
    s = split_days(a)
    assert len(s["test"]) == 14 and len(s["tune"]) == 14
    assert max(s["train"]) < min(s["tune"]) < min(s["test"])
    assert not set(s["train"]) & set(s["tune"])
    assert not set(s["tune"]) & set(s["test"])
    assert s["test"][-1] == usable_origins(a)[-1]  # holdout is the latest period


def test_train_window_is_the_last_n_days_before_the_origin():
    a = alerts_frame()
    u = usable_origins(a)
    end = split_days(a)["tune_start"]
    w = train_window(u, end, days=180)
    assert len(w) == 180
    assert max(w) < end
    assert min(w) >= end - pd.Timedelta(days=180)  # DST can shift the first origin by up to 1 h
    assert min(w) < end - pd.Timedelta(days=179)


def test_match_events_finds_20min_away_only_at_30():
    t = pd.Timestamp("2026-01-01 12:00", tz="UTC")
    pred, actual = [t], [t + pd.Timedelta(minutes=20)]
    assert match_events(pred, actual, 30) == (1, 0, 0)
    assert match_events(pred, actual, 15) == (0, 1, 1)


def test_match_events_edge_cases():
    t = pd.Timestamp("2026-01-01 12:00", tz="UTC")
    assert match_events([], [t], 30) == (0, 0, 1)
    assert match_events([t], [], 30) == (0, 1, 0)
    # two predictions, one actual -> the far prediction is a false positive
    tp, fp, fn = match_events([t, t + pd.Timedelta("3h")], [t], 30)
    assert (tp, fp, fn) == (1, 1, 0)


def test_conformal_width_is_the_hand_computed_quantile():
    pred = np.zeros(5)
    actual = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    assert conformal_width(pred, actual, 0.8) == pytest.approx(0.42)
    assert conformal_width(pred, actual, 0.5) == pytest.approx(0.30)


def test_coverage_counts_points_inside_the_interval():
    pred = np.zeros(4)
    actual = np.array([0.1, -0.2, 0.3, 5.0])
    assert coverage(pred, actual, 0.3) == pytest.approx(0.75)


def test_prf_handles_zero_division():
    assert prf(0, 0, 0) == (0.0, 0.0, 0.0)
    assert prf(1, 0, 0) == (1.0, 1.0, 1.0)
    assert prf(1, 1, 1) == pytest.approx((0.5, 0.5, 0.5))


def test_seasonal_naive_daily_reads_the_week_before():
    hours = pd.date_range("2026-01-08", periods=48, freq="h", tz="UTC")
    history = pd.DataFrame({"region": "A", "hour": hours, "y_start": ([1, 0] * 24), "y_minutes": 10.0})
    frame = pd.DataFrame({
        "origin": pd.Timestamp("2026-01-15", tz="UTC"),
        "region": "A",
        "hour": pd.date_range("2026-01-15", periods=24, freq="h", tz="UTC"),
    })
    # 2026-01-15 - 7d = 2026-01-08, which is in history: 12 alternating starts, 24 x 10 minutes
    assert seasonal_naive_daily(history, frame, "y_start")["pred"].iloc[0] == 12
    assert seasonal_naive_daily(history, frame, "y_minutes")["pred"].iloc[0] == 240.0


def test_folds_never_reach_the_holdout_or_tune():
    a = alerts_frame()
    s = split_days(a)
    folds = rolling_folds(a, n_folds=3)
    assert [f["fold"] for f in folds] == [0, 1, 2]
    for f in folds:
        assert len(f["eval"]) == 14
        assert max(f["eval"]) < min(s["tune"])  # folds end before tune
        assert max(f["eval"]) < min(s["test"])  # and certainly before the holdout
        assert max(f["train"]) < min(f["eval"])  # train strictly precedes eval
        assert len(f["train"]) == 180
    # folds step back 14 days and never overlap
    assert max(folds[1]["eval"]) == min(folds[0]["eval"]) - pd.Timedelta("24h")
    assert not set(folds[0]["eval"]) & set(folds[1]["eval"])
