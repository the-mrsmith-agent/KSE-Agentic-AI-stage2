"""Download the alert snapshot + Open-Meteo weather for modelling regions.

The only network entry point in the project. Idempotent: unchanged downloads are not
rewritten, and a region's weather file is skipped when it already reaches the requested
end date. Always rewrites `data/raw/manifest.json` with provenance for every dataset.

ponytail: daily snapshot CSV; upgrade path: alerts.in.ua API client when a token exists.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import tempfile
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timezone

import pandas as pd

RAW = pathlib.Path("data/raw")
WEATHER_DIR = RAW / "weather"

SNAPSHOT_URL = (
    "https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset"
    "/main/datasets/volunteer_data_en.csv"
)
UPSTREAM_API = (
    "https://api.github.com/repos/Vadimkin/ukrainian-air-raid-sirens-dataset/commits/main"
)
WEATHER_URL = (
    "https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
    "&start_date={start}&end_date={end}"
    "&hourly=temperature_2m,cloud_cover,wind_speed_10m,precipitation"
    "&daily=sunrise,sunset&timezone=UTC"
)
START_DATE = "2022-02-25"

# Oblast centroids, keyed by the exact region label in the snapshot (Luhanska excluded:
# 3 stray rows, no usable series).
REGIONS = {
    "Cherkaska oblast": (49.44, 32.06),
    "Chernihivska oblast": (51.50, 31.29),
    "Chernivetska oblast": (48.29, 25.94),
    "Dnipropetrovska oblast": (48.46, 35.05),
    "Donetska oblast": (48.02, 37.80),
    "Ivano-Frankivska oblast": (48.92, 24.71),
    "Kharkivska oblast": (49.99, 36.23),
    "Khersonska oblast": (46.64, 32.61),
    "Khmelnytska oblast": (49.42, 26.99),
    "Kirovohradska oblast": (48.51, 32.26),
    "Kyiv City": (50.45, 30.52),
    "Kyivska oblast": (49.80, 30.11),
    "Lvivska oblast": (49.84, 24.03),
    "Mykolaivska oblast": (46.97, 32.00),
    "Odeska oblast": (46.48, 30.73),
    "Poltavska oblast": (49.59, 34.55),
    "Rivnenska oblast": (50.62, 26.25),
    "Sumska oblast": (50.91, 34.80),
    "Ternopilska oblast": (49.55, 25.60),
    "Vinnytska oblast": (49.23, 28.47),
    "Volynska oblast": (50.75, 25.34),
    "Zakarpatska oblast": (48.62, 22.29),
    "Zaporizka oblast": (47.84, 35.14),
    "Zhytomyrska oblast": (50.25, 28.66),
}

WEATHER_COLUMNS = [
    "time",
    "temperature_2m",
    "cloud_cover",
    "wind_speed_10m",
    "precipitation",
    "sunrise",
    "sunset",
]


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_bytes(url: str, headers: dict | None = None, attempts: int = 4) -> bytes:
    """GET `url` into memory, retrying with backoff.

    Public endpoints (GitHub, Open-Meteo) reset connections under bursts, so a bare
    request intermittently dies mid-download.
    """
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, headers=headers or {})
            with urllib.request.urlopen(req, timeout=90) as resp:
                return resp.read()
        except (urllib.error.URLError, TimeoutError):
            if i == attempts - 1:
                raise
            time.sleep(5 * 2**i)


def download(url: str, dest: pathlib.Path, headers: dict | None = None) -> str:
    """Fetch `url` into `dest`, returning its sha256. Leaves `dest` untouched when identical."""
    dest = pathlib.Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    data = fetch_bytes(url, headers)
    new_sha = hashlib.sha256(data).hexdigest()
    if dest.exists() and sha256(dest) == new_sha:
        return new_sha
    fd, tmp = tempfile.mkstemp(dir=dest.parent, suffix=".tmp")
    os.close(fd)
    try:
        pathlib.Path(tmp).write_bytes(data)
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return new_sha


def snapshot_summary(path: pathlib.Path) -> dict:
    """Row count and started_at range of the alert snapshot."""
    ts = pd.to_datetime(pd.read_csv(path, usecols=["started_at"])["started_at"], utc=True)
    return {
        "rows": int(len(ts)),
        "started_at_min": ts.min().isoformat(),
        "started_at_max": ts.max().isoformat(),
    }


def slugify(region: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", region.lower()).strip("_")


def fetch_weather(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    url = WEATHER_URL.format(lat=lat, lon=lon, start=start, end=end)
    payload = json.loads(fetch_bytes(url))
    hourly = payload["hourly"]
    df = pd.DataFrame(
        {
            "time": pd.to_datetime(hourly["time"], utc=True),
            "temperature_2m": hourly["temperature_2m"],
            "cloud_cover": hourly["cloud_cover"],
            "wind_speed_10m": hourly["wind_speed_10m"],
            "precipitation": hourly["precipitation"],
        }
    )
    daily = payload["daily"]
    by_day = pd.DataFrame(
        {
            "date": pd.to_datetime(daily["time"], utc=True).normalize(),
            "sunrise": pd.to_datetime(daily["sunrise"], utc=True),
            "sunset": pd.to_datetime(daily["sunset"], utc=True),
        }
    ).set_index("date")
    # Every hourly row inherits its own calendar day's sunrise/sunset (forward-filled across
    # any missing day at the range edges).
    day = df["time"].dt.normalize()
    df["sunrise"] = day.map(by_day["sunrise"]).ffill()
    df["sunset"] = day.map(by_day["sunset"]).ffill()
    return df[WEATHER_COLUMNS]


def weather_covers(path: pathlib.Path, end: str) -> bool:
    if not path.exists():
        return False
    last = pd.read_csv(path, usecols=["time"])["time"].iloc[-1]
    return pd.to_datetime(last, utc=True).date() >= date.fromisoformat(end)


def write_csv(df: pd.DataFrame, dest: pathlib.Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, suffix=".tmp")
    os.close(fd)
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def upstream_sha() -> str:
    headers = {"User-Agent": "kse-air-raid-alerts"}
    return json.loads(fetch_bytes(UPSTREAM_API, headers))["sha"]


def fetch_snapshot(fetched_at: str) -> dict:
    dest = RAW / "alerts_snapshot.csv"
    file_sha = download(SNAPSHOT_URL, dest)
    entry = {
        "source": "Vadimkin/ukrainian-air-raid-sirens-dataset (volunteer_data_en.csv)",
        "url": SNAPSHOT_URL,
        "query": "daily CI snapshot of per-region alert start/stop events, 2022-02-25 → now",
        "fetched_at": fetched_at,
        "upstream_sha": upstream_sha(),
        "path": str(dest),
        "sha256": file_sha,
    }
    entry.update(snapshot_summary(dest))
    return entry


def fetch_all_weather(start: str, end: str, fetched_at: str) -> dict:
    per_region = {}
    rows = 0
    t_min = t_max = None
    for region, (lat, lon) in REGIONS.items():
        path = WEATHER_DIR / f"{slugify(region)}.csv"
        if not weather_covers(path, end):
            write_csv(fetch_weather(lat, lon, start, end), path)
            time.sleep(1)  # spacing requests keeps us clear of Open-Meteo's burst limit
        ts = pd.to_datetime(pd.read_csv(path, usecols=["time"])["time"], utc=True)
        lo, hi = ts.min(), ts.max()
        rows += len(ts)
        t_min = lo if t_min is None else min(t_min, lo)
        t_max = hi if t_max is None else max(t_max, hi)
        per_region[region] = {
            "path": str(path),
            "rows": int(len(ts)),
            "time_min": lo.isoformat(),
            "time_max": hi.isoformat(),
        }
    return {
        "source": "Open-Meteo historical weather archive (ERA5)",
        "url_template": WEATHER_URL,
        "query": (
            f"hourly temperature_2m,cloud_cover,wind_speed_10m,precipitation + "
            f"daily sunrise,sunset at region centroids; {start} → {end}; timezone=UTC"
        ),
        "fetched_at": fetched_at,
        "start_date": start,
        "end_date": end,
        "region_count": len(REGIONS),
        "rows": rows,
        "time_min": t_min.isoformat(),
        "time_max": t_max.isoformat(),
        "regions": per_region,
    }


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    fetched_at = datetime.now(timezone.utc).isoformat()
    end = datetime.now(timezone.utc).date().isoformat()
    manifest = {
        "generated_at": fetched_at,
        "alerts_snapshot": fetch_snapshot(fetched_at),
        "weather": fetch_all_weather(START_DATE, end, fetched_at),
    }
    (RAW / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    snap = manifest["alerts_snapshot"]
    wx = manifest["weather"]
    print(f"alerts: {snap['rows']} rows {snap['started_at_min']} → {snap['started_at_max']}")
    print(f"weather: {wx['region_count']} regions, {wx['rows']} rows {wx['time_min']} → {wx['time_max']}")


if __name__ == "__main__":
    main()
