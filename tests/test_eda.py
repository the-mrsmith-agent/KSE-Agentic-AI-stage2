"""Self-checks for src/eda.py. Offline synthetic fixtures only: no data/raw, no network."""

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.eda import (
    burden,
    censored_share,
    daily_starts,
    gap_hours,
    hour_of_week_share,
    regime_break,
)


def alerts_fixture(rows):
    """rows: (region, start_iso_utc, minutes, naive) -> frame shaped like load_alerts()."""
    recs = []
    for region, iso, mins, naive in rows:
        start = pd.Timestamp(iso, tz="UTC")
        recs.append(
            {
                "region": region,
                "started_at": start,
                "finished_at": start + pd.Timedelta(minutes=mins),
                "naive": naive,
                # features.py sets duration_min to NaN for naive/right-censored rows
                "duration_min": np.nan if naive else float(mins),
            }
        )
    return pd.DataFrame(recs)


def test_gap_hours_reports_known_three_hour_gap():
    a = alerts_fixture(
        [
            ("R", "2024-01-01 00:00:00+00:00", 30, False),
            ("R", "2024-01-01 03:00:00+00:00", 30, False),
        ]
    )
    assert gap_hours(a).tolist() == [3.0]


def test_gap_hours_are_per_region():
    a = alerts_fixture(
        [
            ("A", "2024-01-01 00:00:00+00:00", 30, False),
            ("B", "2024-01-01 00:00:00+00:00", 30, False),
            ("A", "2024-01-01 02:00:00+00:00", 30, False),
            ("B", "2024-01-01 05:00:00+00:00", 30, False),
        ]
    )
    assert sorted(gap_hours(a)) == [2.0, 5.0]


def test_burden_two_thirty_minute_alerts():
    a = alerts_fixture(
        [
            ("R", "2024-01-01 00:00:00+00:00", 30, False),
            ("R", "2024-01-01 01:00:00+00:00", 30, False),
        ]
    )
    b = burden(a)
    assert b.loc["R", "alert_minutes"] == 60.0
    assert b.loc["R", "starts"] == 2
    assert b.loc["R", "censored_share"] == 0.0


def test_hour_of_week_share_single_event():
    a = alerts_fixture([("R", "2024-06-10 09:00:00+00:00", 30, False)])  # Mon 12:00 Kyiv (EEST)
    share = hour_of_week_share(a)
    assert share.shape == (7, 24)
    assert share.to_numpy().sum() == 1.0
    assert share.loc[0, 12] == 1.0


def test_daily_starts_sums_on_local_days():
    grid = pd.DataFrame(
        {
            "region": ["R", "R", "R"],
            "hour": pd.to_datetime(
                ["2024-06-10 21:00:00+00:00", "2024-06-10 22:00:00+00:00", "2024-06-11 00:00:00+00:00"],
                utc=True,
            ),
            "minutes": [10.0, 20.0, 5.0],
            "starts": [1, 0, 1],
        }
    )
    d = daily_starts(grid).set_index("date")
    # 21:00 & 22:00 UTC are 2024-06-11 00:00/01:00 Kyiv; 00:00 UTC is 03:00 Kyiv
    assert d.loc[pd.Timestamp("2024-06-11"), "starts"] == 2
    assert d.loc[pd.Timestamp("2024-06-11"), "minutes"] == 35.0


def test_regime_break_detects_upward_jump():
    dates = pd.date_range("2024-01-01", periods=120, freq="D")
    starts = [1.0] * 60 + [10.0] * 60
    brk = regime_break(pd.DataFrame({"date": dates, "starts": starts}))
    assert brk["after"] > brk["before"]
    assert brk["before"] == 1.0
    assert brk["after"] == 10.0
    assert abs(dates.get_loc(brk["date"]) - 59) <= 1


def test_censored_share_counts_naive_rows():
    a = alerts_fixture(
        [
            ("R", "2024-01-01 00:00:00+00:00", 30, False),
            ("R", "2024-01-01 01:00:00+00:00", 30, True),
        ]
    )
    c = censored_share(a)
    assert c["naive"] == 0.5
    assert c["duration"] == 0.5
