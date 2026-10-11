"""Origin-guarded feature + label builder for air-raid alert forecasting.

`build(origin)` is the single feature path shared by training and evaluation. It returns
one row per region × forecast hour (origin + 0..23 h, UTC) whose features are derived only
from data strictly before `origin`, plus the labels (`y_*`), which may look at/after the
origin and therefore never pass through the leakage guard.

Every feature family slices its source to `< origin` and asserts the slice is leak-free
through one shared helper, `_guard`. Calendar and sunrise/sunset are deliberately outside
the guard: they are computable years ahead, so reading them for the horizon is not a leak.

ponytail: weather enters as past-24h lags; upgrade path: read the Open-Meteo forecast API
at origin. ponytail: fixed month-day UA holidays; upgrade path: Orthodox Easter/Trinity.
"""

from __future__ import annotations

import functools
import pathlib
import zoneinfo

import numpy as np
import pandas as pd

from src.fetch import REGIONS, slugify

KYIV = zoneinfo.ZoneInfo("Europe/Kyiv")

DEFAULT_ALERTS = "data/raw/alerts_snapshot.csv"
DEFAULT_WEATHER = "data/raw/weather"

# Fixed Ukrainian public holidays as month-day (no Easter/Trinity, which move with the
# Orthodox calendar). Kept as month-day so it needs no yearly maintenance, and it is known
# years ahead — safe to attach to any forecast hour.
HOLIDAYS = {
    "01-01",  # New Year
    "01-07",  # Orthodox Christmas
    "03-08",  # International Women's Day
    "05-01",  # Labour Day
    "05-09",  # Victory Day
    "06-28",  # Constitution Day
    "08-24",  # Independence Day
    "10-14",  # Defenders Day
    "12-25",  # Christmas
}

# Right-censored (naive) or implausibly long alerts keep their start event but drop out of
# the duration label; 24h is safely above any real single alert.
MAX_DURATION_MIN = 24 * 60
# Alerts can be silent for months (long quiet regimes); uncapped "hours since" would let
# the model extrapolate on an outlier. 90 days is well past the point where the signal
# saturates.
MAX_HOURS_SINCE = 24 * 90

SLUG_TO_REGION = {slugify(region): region for region in REGIONS}

FEATURES = [
    "hour_local",
    "day_of_week",
    "is_weekend",
    "month",
    "is_holiday",
    "starts_past_1h",
    "starts_past_6h",
    "starts_past_24h",
    "starts_past_72h",
    "alert_minutes_past_24h",
    "trailing_28d_starts_per_day",
    "hours_since_last_alert",
    "regions_alerting_past_3h",
    "temp_mean_24h",
    "cloud_mean_24h",
    "wind_mean_24h",
    "precip_sum_24h",
    "minutes_to_sunset",
    "is_night",
    "day_length_min",
]


def _as_utc(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _guard(frame: pd.DataFrame, col: str, origin: pd.Timestamp) -> pd.DataFrame:
    """Assert a feature slice holds nothing at/after `origin`. Labels never call this."""
    assert bool((frame[col] < origin).all()), "leak: feature row at/after origin"
    return frame


@functools.lru_cache(maxsize=None)
def _load_alerts_cached(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["region", "started_at", "finished_at", "naive"])
    df["started_at"] = pd.to_datetime(df["started_at"], utc=True)
    df["finished_at"] = pd.to_datetime(df["finished_at"], utc=True)
    df = df[df["region"] != "Luhanska oblast"].copy()  # 3 stray rows, no usable series
    df["duration_min"] = (df["finished_at"] - df["started_at"]).dt.total_seconds() / 60.0
    censored = df["naive"] | (df["duration_min"] > MAX_DURATION_MIN)
    df.loc[censored, "duration_min"] = np.nan
    return df.sort_values(["region", "started_at"]).reset_index(drop=True)


def load_alerts(path: str | pathlib.Path = DEFAULT_ALERTS) -> pd.DataFrame:
    """Cached, read-only alert table: `region, started_at, finished_at, naive, duration_min`."""
    return _load_alerts_cached(str(path))


@functools.lru_cache(maxsize=None)
def load_weather(directory: str | pathlib.Path = DEFAULT_WEATHER) -> pd.DataFrame:
    """Cached, read-only hourly weather for every region (`time` UTC-aware)."""
    frames = []
    for path in sorted(pathlib.Path(directory).glob("*.csv")):
        df = pd.read_csv(path)
        for col in ("time", "sunrise", "sunset"):
            df[col] = pd.to_datetime(df[col], utc=True)
        df["region"] = SLUG_TO_REGION.get(path.stem, path.stem)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["region", "time"]).reset_index(drop=True)


def hourly_grid(alerts: pd.DataFrame) -> pd.DataFrame:
    """Dense `region × hour` grid: alert-minutes overlapping each hour and starts counts.

    The single source behind every lag and label: minutes split alert-seconds across the
    hour buckets it spans, starts counts alerts whose `started_at` floors into the hour.
    """
    a = alerts[["region", "started_at", "finished_at"]].reset_index(drop=True)

    def naive(s: pd.Series) -> np.ndarray:
        return s.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()

    start, finish = naive(a["started_at"]), naive(a["finished_at"])
    start_h = a["started_at"].dt.floor("h")
    end_h = a["finished_at"].dt.floor("h")
    n = (((end_h - start_h).dt.total_seconds() // 3600).astype("int64") + 1).clip(lower=1)
    n_arr = n.to_numpy()
    idx = np.repeat(np.arange(len(a)), n_arr)
    offs = np.arange(len(idx)) - np.repeat(np.cumsum(n_arr) - n_arr, n_arr)
    hour = naive(start_h)[idx] + offs * np.timedelta64(1, "h")
    minutes = (np.minimum(finish[idx], hour + np.timedelta64(1, "h")) - np.maximum(start[idx], hour))
    minutes = np.clip(minutes / np.timedelta64(1, "m"), 0.0, None)

    agg = pd.DataFrame(
        {"region": a["region"].to_numpy()[idx], "hour": hour, "minutes": minutes}
    ).groupby(["region", "hour"], as_index=False)["minutes"].sum()
    agg["hour"] = pd.to_datetime(agg["hour"], utc=True)
    starts = (
        a.assign(hour=start_h)
        .groupby(["region", "hour"], as_index=False)
        .size()
        .rename(columns={"size": "starts"})
    )

    regions = sorted(a["region"].unique())
    hours = pd.date_range(start_h.min(), end_h.max(), freq="h")
    grid = pd.DataFrame({"region": regions}).merge(pd.DataFrame({"hour": hours}), how="cross")
    grid = grid.merge(agg, on=["region", "hour"], how="left").merge(starts, on=["region", "hour"], how="left")
    grid["minutes"] = grid["minutes"].fillna(0.0)
    grid["starts"] = grid["starts"].fillna(0).astype("int64")
    return grid.sort_values(["region", "hour"]).reset_index(drop=True)


@functools.lru_cache(maxsize=None)
def load_grid(path: str | pathlib.Path = DEFAULT_ALERTS) -> pd.DataFrame:
    """Cached hourly grid; keeps ~90 backtest origins from rebuilding ~1M grid rows."""
    return hourly_grid(load_alerts(path))


def daily_origin(day) -> pd.Timestamp:
    """00:00 Europe/Kyiv on `day` (str/date/Timestamp) as a UTC-aware Timestamp."""
    ts = pd.Timestamp(day)
    ts = ts.tz_localize(KYIV) if ts.tzinfo is None else ts.tz_convert(KYIV)
    return ts.normalize().tz_convert("UTC")


def origins(start_day, end_day) -> list[pd.Timestamp]:
    """Inclusive daily origins over `[start_day, end_day]` (Europe/Kyiv midnight, UTC)."""
    days = pd.date_range(pd.Timestamp(start_day).date(), pd.Timestamp(end_day).date(), freq="D")
    return [daily_origin(day) for day in days]


def _lags(alerts: pd.DataFrame, grid: pd.DataFrame, origin: pd.Timestamp, regions: list[str]) -> pd.DataFrame:
    past = _guard(grid[grid["hour"] < origin].copy(), "hour", origin)
    past_alerts = _guard(alerts[alerts["started_at"] < origin].copy(), "started_at", origin)

    out = pd.DataFrame(index=pd.Index(regions, name="region"))
    for name, span in (
        ("starts_past_1h", "1h"),
        ("starts_past_6h", "6h"),
        ("starts_past_24h", "24h"),
        ("starts_past_72h", "72h"),
    ):
        window = past[past["hour"] >= origin - pd.Timedelta(span)]
        out[name] = window.groupby("region")["starts"].sum().reindex(out.index, fill_value=0)

    window = past[past["hour"] >= origin - pd.Timedelta("24h")]
    out["alert_minutes_past_24h"] = window.groupby("region")["minutes"].sum().reindex(out.index, fill_value=0.0)

    window = past[past["hour"] >= origin - pd.Timedelta("28D")]
    out["trailing_28d_starts_per_day"] = (
        window.groupby("region")["starts"].sum().reindex(out.index, fill_value=0) / 28.0
    )

    last = past_alerts.groupby("region")["started_at"].max()
    since = (origin - last).dt.total_seconds() / 3600.0
    out["hours_since_last_alert"] = since.reindex(out.index).clip(upper=MAX_HOURS_SINCE)

    # Mass attacks are nationwide, so how many regions are alerting right now leads each region.
    window = past[past["hour"] >= origin - pd.Timedelta("3h")]
    out["regions_alerting_past_3h"] = int(window.loc[window["minutes"] > 0, "region"].nunique())
    return out.reset_index()


def _weather_lags(weather: pd.DataFrame, origin: pd.Timestamp, regions: list[str]) -> pd.DataFrame:
    past = _guard(weather[weather["time"] < origin].copy(), "time", origin)
    tail = past.sort_values(["region", "time"]).groupby("region").tail(24)
    agg = tail.groupby("region").agg(
        temp_mean_24h=("temperature_2m", "mean"),
        cloud_mean_24h=("cloud_cover", "mean"),
        wind_mean_24h=("wind_speed_10m", "mean"),
        precip_sum_24h=("precipitation", "sum"),
    )
    return agg.reindex(regions).reset_index()


def _daylight(weather: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Minutes-to-sunset / night flag / day length per forecast hour.

    Sunrise and sunset are deterministic astronomy known years ahead, so this family is
    intentionally NOT routed through `_guard` even though the weather file carries them.
    """
    day = (
        weather.assign(date=weather["time"].dt.tz_convert("UTC").dt.normalize())
        .groupby(["region", "date"], as_index=False)[["sunrise", "sunset"]]
        .first()
        .sort_values(["region", "date"])
    )
    day[["sunrise", "sunset"]] = day.groupby("region")[["sunrise", "sunset"]].ffill()

    out = rows[["region", "hour"]].copy()
    out["date"] = out["hour"].dt.normalize()
    out = out.merge(day, on=["region", "date"], how="left")
    for col in ("sunrise", "sunset"):
        tod = out[col].dt.hour * 60 + out[col].dt.minute + out[col].dt.second / 60.0
        median_tod = tod.groupby(out["region"]).transform("median")  # fallback for gap dates
        out[col] = out[col].fillna(out["date"] + pd.to_timedelta(median_tod, unit="m"))

    return pd.DataFrame(
        {
            "region": out["region"],
            "hour": out["hour"],
            "minutes_to_sunset": (out["sunset"] - out["hour"]).dt.total_seconds() / 60.0,
            "is_night": ((out["hour"] < out["sunrise"]) | (out["hour"] > out["sunset"])).astype(int),
            "day_length_min": (out["sunset"] - out["sunrise"]).dt.total_seconds() / 60.0,
        }
    )


def _labels(alerts: pd.DataFrame, grid: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Start flag, alert-minutes and log-duration-at-start per hour. Never guarded."""
    out = rows[["region", "hour"]].merge(
        grid[["region", "hour", "minutes", "starts"]].rename(columns={"minutes": "y_minutes", "starts": "y_start"}),
        on=["region", "hour"],
        how="left",
    )
    out["y_minutes"] = out["y_minutes"].fillna(0.0)
    out["y_start"] = out["y_start"].fillna(0).gt(0).astype(int)

    first = (
        alerts.assign(hour=alerts["started_at"].dt.floor("h"))
        .sort_values(["region", "hour", "started_at"])
        .drop_duplicates(["region", "hour"], keep="first")
        .assign(y_dur_log=lambda d: np.log1p(d["duration_min"]))  # NaN when censored/>24h
    )
    return out.merge(first[["region", "hour", "y_dur_log"]], on=["region", "hour"], how="left")


def build(
    origin: pd.Timestamp,
    alerts: pd.DataFrame | None = None,
    weather: pd.DataFrame | None = None,
    grid: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Feature matrix for one origin: 24 region × hour rows with lags and labels.

    Pass nothing for the fast cached path (backtests); pass fixtures for tests. All loaded
    tables are treated as read-only.
    """
    origin = _as_utc(origin)
    if alerts is None and grid is None:
        grid = load_grid()
    alerts = load_alerts() if alerts is None else alerts
    weather = load_weather() if weather is None else weather
    grid = hourly_grid(alerts) if grid is None else grid

    regions = sorted(alerts["region"].unique())
    hours = pd.date_range(origin, periods=24, freq="h")
    rows = pd.DataFrame({"region": regions}).merge(pd.DataFrame({"hour": hours}), how="cross")

    local = rows["hour"].dt.tz_convert(KYIV)
    rows["hour_local"] = local.dt.hour.astype(int)
    rows["day_of_week"] = local.dt.dayofweek.astype(int)
    rows["is_weekend"] = (local.dt.dayofweek >= 5).astype(int)
    rows["month"] = local.dt.month.astype(int)
    rows["is_holiday"] = local.dt.strftime("%m-%d").isin(HOLIDAYS).astype(int)

    out = (
        rows.merge(_lags(alerts, grid, origin, regions), on="region", how="left")
        .merge(_weather_lags(weather, origin, regions), on="region", how="left")
        .merge(_daylight(weather, rows), on=["region", "hour"], how="left")
        .merge(_labels(alerts, grid, rows), on=["region", "hour"], how="left")
    )
    out.insert(0, "origin", origin)
    out = out[["origin", "region", "hour", *FEATURES, "y_start", "y_minutes", "y_dur_log"]]
    return out.sort_values(["region", "hour"]).reset_index(drop=True)
