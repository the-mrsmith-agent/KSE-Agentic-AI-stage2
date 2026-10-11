"""Self-checks for src/features.py. Offline synthetic fixtures only: no data/raw, no network."""

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.features import FEATURES, _guard, build, daily_origin, hourly_grid, load_alerts

REGION = "Test oblast"
ORIGIN = daily_origin("2024-06-10")  # 2024-06-09 21:00 UTC (EEST)


def write_alerts(tmp_path, rows):
    """rows: (region, started_at_iso, finished_at_iso, naive). Returns load_alerts output."""
    path = tmp_path / "alerts.csv"
    lines = ["region,started_at,finished_at,naive"] + [f"{r[0]},{r[1]},{r[2]},{r[3]}" for r in rows]
    path.write_text("\n".join(lines) + "\n")
    return load_alerts(path)


def weather_fixture(region=REGION, start="2024-06-07", hours=120):
    times = pd.date_range(start, periods=hours, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "region": region,
            "time": times,
            "temperature_2m": 10.0,
            "cloud_cover": 50.0,
            "wind_speed_10m": 3.0,
            "precipitation": 0.0,
            "sunrise": times.normalize() + pd.Timedelta(hours=4),
            "sunset": times.normalize() + pd.Timedelta(hours=18),
        }
    )


def test_guard_fires_on_alert_at_origin():
    at_origin = pd.DataFrame({"started_at": pd.to_datetime([ORIGIN], utc=True)})
    with pytest.raises(AssertionError, match="leak"):
        _guard(at_origin, "started_at", ORIGIN)
    # the correctly sliced source is accepted and empty
    assert _guard(at_origin[at_origin["started_at"] < ORIGIN], "started_at", ORIGIN).empty


def test_load_alerts_drops_luhansk_and_censors_durations(tmp_path):
    alerts = write_alerts(
        tmp_path,
        [
            ("Kyiv City", "2024-01-01 00:00:00+00:00", "2024-01-01 01:00:00+00:00", False),
            ("Luhanska oblast", "2024-01-01 00:00:00+00:00", "2024-01-01 01:00:00+00:00", False),
            ("Kyiv City", "2024-01-02 00:00:00+00:00", "2024-01-03 06:00:00+00:00", False),  # 30 h
            ("Kyiv City", "2024-01-03 00:00:00+00:00", "2024-01-03 00:30:00+00:00", True),  # censored
        ],
    )
    assert set(alerts["region"]) == {"Kyiv City"}
    assert alerts["duration_min"].isna().sum() == 2  # the 30 h and the naive row
    assert alerts.loc[alerts["duration_min"].notna(), "duration_min"].tolist() == [60.0]


def test_hourly_grid_splits_minutes_across_hours(tmp_path):
    alerts = write_alerts(
        tmp_path, [("Kyiv City", "2024-01-01 10:36:00+00:00", "2024-01-01 11:06:00+00:00", False)]
    )
    grid = hourly_grid(alerts).set_index(["region", "hour"])
    assert grid.loc[("Kyiv City", pd.Timestamp("2024-01-01 10:00", tz="UTC")), "minutes"] == 24.0
    assert grid.loc[("Kyiv City", pd.Timestamp("2024-01-01 11:00", tz="UTC")), "minutes"] == 6.0
    assert grid["starts"].sum() == 1


def test_lag_features(tmp_path):
    start = ORIGIN - pd.Timedelta(hours=2)
    alerts = write_alerts(
        tmp_path, [(REGION, start.isoformat(), (start + pd.Timedelta(minutes=30)).isoformat(), False)]
    )
    rows = build(ORIGIN, alerts=alerts, weather=weather_fixture())

    assert len(rows) == 24 and set(rows["region"]) == {REGION}
    assert rows["hour"].tolist() == list(pd.date_range(ORIGIN, periods=24, freq="h"))
    row = rows.iloc[0]
    assert row["starts_past_24h"] == 1
    assert row["alert_minutes_past_24h"] == 30.0
    assert row["hours_since_last_alert"] == 2.0
    assert row["starts_past_1h"] == 0
    assert list(rows.columns) == ["origin", "region", "hour", *FEATURES, "y_start", "y_minutes", "y_dur_log"]


def test_label_with_known_duration(tmp_path):
    start = ORIGIN + pd.Timedelta(hours=3)
    alerts = write_alerts(
        tmp_path, [(REGION, start.isoformat(), (start + pd.Timedelta(minutes=30)).isoformat(), False)]
    )
    rows = build(ORIGIN, alerts=alerts, weather=weather_fixture())
    hour = rows[(rows["region"] == REGION) & (rows["hour"] == start)].iloc[0]
    assert hour["y_start"] == 1
    assert hour["y_minutes"] == 30.0
    assert hour["y_dur_log"] == pytest.approx(np.log1p(30))


def test_label_censored_start_has_nan_duration(tmp_path):
    start = ORIGIN + pd.Timedelta(hours=3)
    alerts = write_alerts(
        tmp_path, [(REGION, start.isoformat(), (start + pd.Timedelta(minutes=30)).isoformat(), True)]
    )
    rows = build(ORIGIN, alerts=alerts, weather=weather_fixture())
    hour = rows[(rows["region"] == REGION) & (rows["hour"] == start)].iloc[0]
    assert hour["y_start"] == 1
    assert np.isnan(hour["y_dur_log"])
