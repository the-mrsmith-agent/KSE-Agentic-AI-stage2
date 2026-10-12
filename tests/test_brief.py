"""Self-check for src.brief: the briefing renders, and it stays free of jargon.

The jargon list is the point of the module, so it is asserted rather than trusted: if a
statistics term leaks back into a template, this fails.
"""

from __future__ import annotations

import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import src.brief as brief
from src.brief import build

BANNED = [
    "mae", "rmse", "f1", "brier", "conformal", "holdout", "leakage", "baseline",
    "stationar", "poisson", "histgb", "logistic", "quantile", "seasonal naive",
    "precision", "recall", "coverage (%)",
]


def _alerts() -> pd.DataFrame:
    rows = [
        # region, started_at, finished_at, naive
        ("A oblast", "2026-01-01 20:00:00+00:00", "2026-01-01 20:40:00+00:00", False),
        ("A oblast", "2026-01-02 08:00:00+00:00", "2026-01-02 09:00:00+00:00", False),
        ("A oblast", "2026-01-05 23:00:00+00:00", "2026-01-05 23:30:00+00:00", True),
        ("B oblast", "2026-03-01 12:00:00+00:00", "2026-03-01 12:30:00+00:00", False),
        ("B oblast", "2026-03-08 12:00:00+00:00", "2026-03-08 13:30:00+00:00", False),
    ]
    df = pd.DataFrame(rows, columns=["region", "started_at", "finished_at", "naive"])
    for col in ("started_at", "finished_at"):
        df[col] = pd.to_datetime(df[col], utc=True)
    df["duration_min"] = (df["finished_at"] - df["started_at"]).dt.total_seconds() / 60
    df.loc[df["naive"], "duration_min"] = None  # censored ends carry no duration
    return df


def _metrics() -> dict:
    return {
        "splits": {"test": ["2026-09-26T21:00:00+00:00", "2026-10-09T21:00:00+00:00", 14]},
        "counts_daily": {
            "y_starts_day": {"histgb": {"mae": 1.5, "national_mae": 11.0},
                             "seasonal_naive": {"mae": 2.2, "national_mae": 22.0}}
        },
        "start_events": {
            "histgb": {"tol_30": {"precision": 0.5, "recall": 0.5}},
            "seasonal_naive": {"tol_30": {"precision": 0.4, "recall": 0.4}},
        },
        "duration": {
            "histgb_q50": {"mae_minutes": 20.0, "coverage": 0.77},
            "duration_naive": {"mae_minutes": 21.0},
        },
    }


def _preds() -> pd.DataFrame:
    return pd.DataFrame({
        "origin": pd.to_datetime(["2026-09-26T21:00:00+00:00", "2026-09-27T21:00:00+00:00"], utc=True),
        "region": ["A oblast", "B oblast"],
        "y_starts_day": [3.0, 5.0],
        "y_minutes_day": [60.0, 120.0],
        "p_histgb_start": [2.5, 4.5],
        "p_snaive_start": [1.0, 6.0],
        "p_histgb_min": [50.0, 130.0],
    })


def test_brief_renders_with_forecast_and_without():
    text = build(_alerts(), _metrics(), _preds())
    assert "A oblast" in text and "B oblast" in text
    assert "1.5 alerts" in text  # counts verdict, in plain units
    assert "| Day |" in text  # day-by-day reality vs guess table
    assert "77 times in 100" in text  # coverage, translated

    stripped = build(_alerts(), None, None)
    assert "python -m src.evaluate" in stripped  # graceful fallback
    assert "A oblast" in stripped


def test_brief_embeds_figures_only_when_present(tmp_path, monkeypatch):
    monkeypatch.setattr(brief, "FIGDIR", tmp_path)
    assert brief._fig("daily_starts.png", "caption") == []  # nothing generated yet

    (tmp_path / "daily_starts.png").write_bytes(b"not a real png")
    line = brief._fig("daily_starts.png", "Alert starts per day")
    assert line == ["![Alert starts per day](figures/daily_starts.png)", ""]


def test_brief_has_no_jargon():
    text = build(_alerts(), _metrics(), _preds()).lower()
    for word in BANNED:
        assert word not in text, f"jargon leaked into the brief: {word!r}"
