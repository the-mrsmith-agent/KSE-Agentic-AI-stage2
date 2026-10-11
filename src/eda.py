"""Exploratory analysis of Ukrainian air-raid alerts: seasonality, clustering, gaps,
regime shift and per-oblast burden. Writes figures to `reports/figures/` and the numbers
to `reports/eda.md`.

Every helper is pure and takes the frames from `src.features`, so the whole module is
reproducible from `python -m src.eda` and testable offline. Windows are Europe/Kyiv local
(the forecast origin convention) unless a column is explicitly UTC.

ponytail: single hard-coded 28-day window for the change point; upgrade path: sweep the
window and report a stability band.
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")  # non-interactive: must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.features import KYIV, hourly_grid, load_alerts

FIGDIR = pathlib.Path("reports/figures")
REPORT = pathlib.Path("reports/eda.md")
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def daily_starts(grid: pd.DataFrame) -> pd.DataFrame:
    """National daily alert-starts and alert-minutes from the dense region×hour grid.

    Days are Europe/Kyiv calendar days, matching the forecast origin convention.
    """
    g = grid.copy()
    g["date"] = g["hour"].dt.tz_convert(KYIV).dt.normalize().dt.tz_localize(None)
    out = g.groupby("date").agg(starts=("starts", "sum"), minutes=("minutes", "sum"))
    return out.sort_index().reset_index()


def hour_of_week_share(alerts: pd.DataFrame) -> pd.DataFrame:
    """7×24 share of national alert-starts by local day-of-week (Mon=0) × local hour."""
    local = alerts["started_at"].dt.tz_convert(KYIV)
    counts = pd.crosstab(
        local.dt.dayofweek.astype(int), local.dt.hour.astype(int)
    ).reindex(index=range(7), columns=range(24), fill_value=0)
    return counts / counts.to_numpy().sum()


def gap_hours(alerts: pd.DataFrame) -> pd.Series:
    """Hours between consecutive alert-starts within each region (quiet gaps)."""
    a = alerts.sort_values(["region", "started_at"])
    return (a.groupby("region")["started_at"].diff().dt.total_seconds() / 3600.0).dropna()


def burden(alerts: pd.DataFrame) -> pd.DataFrame:
    """Per-region burden: alert count, alert-minutes, alerts per 100 days.

    Alert-minutes sum `duration_min`, which is NaN for right-censored and >24h alerts, so
    those contribute 0 minutes but still count as a start (see `censored_share`).
    """
    a = alerts.sort_values(["region", "started_at"])
    span_days = a.groupby("region")["started_at"].agg(
        lambda s: (s.max() - s.min()).total_seconds() / 86400.0
    )
    out = pd.DataFrame(
        {
            "starts": a.groupby("region").size(),
            "alert_minutes": a.groupby("region")["duration_min"].sum(),
            "censored_share": a.groupby("region")["duration_min"].apply(
                lambda s: float(s.isna().mean())
            ),
        }
    )
    out["starts_per_100d"] = out["starts"] / span_days * 100.0
    return out.sort_values("alert_minutes", ascending=False)


def regime_break(daily: pd.DataFrame, window: int = 28) -> dict:
    """Single change point of national daily starts: split maximising next-vs-previous mean.

    Compares the mean of the `window` days after a candidate split with the `window` days
    before it and returns the split with the largest increase.
    """
    x = daily.set_index("date")["starts"].asfreq("D").fillna(0.0)
    before = x.rolling(window).mean()
    after = x.rolling(window).mean().shift(-window)
    diff = after - before
    date = diff.idxmax()
    return {"date": date, "before": float(before.loc[date]), "after": float(after.loc[date])}


def censored_share(alerts: pd.DataFrame) -> dict:
    """Share of starts that are `naive` (end never arrived) vs. duration-censored (incl. >24h)."""
    return {
        "naive": float(alerts["naive"].mean()),
        "duration": float(alerts["duration_min"].isna().mean()),
    }


def year_rate(daily: pd.DataFrame) -> pd.DataFrame:
    """Starts-per-day by calendar year (fair across partial years like the current one)."""
    y = daily.assign(year=daily["date"].dt.year).groupby("year")
    return (y["starts"].sum() / y.size()).rename("starts_per_day").reset_index()


def _plot_daily(daily: pd.DataFrame, path: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(daily["date"], daily["starts"], lw=0.5, alpha=0.4, label="daily starts")
    ax.plot(
        daily["date"],
        daily["starts"].rolling(28, min_periods=1).mean(),
        lw=1.8,
        color="crimson",
        label="28-day mean",
    )
    ax.set(title="National air-raid alert starts per day", ylabel="starts / day")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_hour_of_week(share: pd.DataFrame, peak: tuple[int, int], path: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 3.6))
    im = ax.imshow(share.to_numpy() * 100.0, aspect="auto", cmap="magma")
    ax.set(
        xticks=range(24),
        xticklabels=[f"{h:02d}" for h in range(24)],
        yticks=range(7),
        yticklabels=WEEKDAYS,
        xlabel="hour (Europe/Kyiv)",
        title="Share of alert starts by local day-of-week × hour (%)",
    )
    ax.set_xticks(np.arange(-0.5, 24, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 7, 1), minor=True)
    ax.grid(which="minor", color="w", lw=0.3)
    ax.tick_params(which="minor", length=0)
    ax.scatter([peak[1]], [peak[0]], s=80, facecolors="none", edgecolors="cyan")
    fig.colorbar(im, ax=ax, label="% of all starts")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_monthly(daily: pd.DataFrame, path: pathlib.Path) -> None:
    m = daily.assign(month=daily["date"].dt.to_period("M").dt.to_timestamp())
    agg = m.groupby("month").agg(starts=("starts", "sum"), minutes=("minutes", "mean"))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(agg.index, agg["starts"], width=20, color="steelblue", label="starts / month")
    ax.set(ylabel="alert starts / month", title="Monthly alert starts and mean alert-minutes per day")
    ax2 = ax.twinx()
    ax2.plot(agg.index, agg["minutes"], color="darkorange", lw=1.8, label="mean min / day")
    ax2.set_ylabel("mean alert-minutes / day", color="darkorange")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_gaps(gaps: pd.Series, since_days: pd.Series, path: pathlib.Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    bins = np.logspace(np.log10(max(gaps.min(), 1e-3)), np.log10(gaps.max()), 60)
    ax1.hist(gaps, bins=bins, color="teal")
    ax1.set_xscale("log")
    ax1.set(xlabel="gap between consecutive alerts (h)", ylabel="count", title="Quiet gaps (log scale)")
    ax2.barh(since_days.sort_values().index, since_days.sort_values().to_numpy(), color="slateblue")
    ax2.set(xlabel="days since last alert (at data end)", title="Current quiet stretch per region")
    ax2.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_burden(b: pd.DataFrame, path: pathlib.Path) -> None:
    s = b.sort_values("alert_minutes")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(s.index, s["alert_minutes"], color="indianred")
    ax.set(xlabel="total known alert-minutes (2022–2026)", title="Per-oblast alert burden")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_durations(alerts: pd.DataFrame, cens: dict, path: pathlib.Path) -> None:
    d = alerts["duration_min"].dropna()
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.logspace(np.log10(0.5), np.log10(d.max() + 1), 50)
    ax.hist(d, bins=bins, color="seagreen")
    ax.set_xscale("log")
    ax.axvline(30, color="crimson", lw=1.2, ls="--")
    ax.text(31, ax.get_ylim()[1] * 0.8, "30 min (censored imputation)", color="crimson", fontsize=8)
    ax.set(
        xlabel="alert duration (min, log scale)",
        ylabel="count",
        title=f"Alert durations — {cens['duration']:.1%} censored (excluded from this histogram)",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def analyse(alerts: pd.DataFrame, grid: pd.DataFrame) -> dict:
    """Compute every number reported in `reports/eda.md`."""
    daily = daily_starts(grid)
    share = hour_of_week_share(alerts)
    peak_dow, peak_hour = np.unravel_index(share.to_numpy().argmax(), share.shape)
    gap = gap_hours(alerts)
    b = burden(alerts)
    cens = censored_share(alerts)
    rate = year_rate(daily)
    brk = regime_break(daily)

    last = (
        alerts.groupby("region")["started_at"]
        .max()
        .dt.tz_convert(KYIV)
        .dt.tz_localize(None)
        .dt.normalize()
    )
    since = (daily["date"].max() - last).dt.total_seconds() / 86400.0
    return {
        "daily": daily,
        "share": share,
        "peak": (int(peak_dow), int(peak_hour)),
        "gaps": gap,
        "burden": b,
        "censored": cens,
        "rate": rate,
        "break": brk,
        "since_days": since,
    }


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    alerts = load_alerts()
    grid = hourly_grid(alerts)
    r = analyse(alerts, grid)

    _plot_daily(r["daily"], FIGDIR / "daily_starts.png")
    _plot_hour_of_week(r["share"], r["peak"], FIGDIR / "hour_of_week_heatmap.png")
    _plot_monthly(r["daily"], FIGDIR / "monthly_burden.png")
    _plot_gaps(r["gaps"], r["since_days"], FIGDIR / "gaps_and_quiet.png")
    _plot_burden(r["burden"], FIGDIR / "region_burden.png")
    _plot_durations(alerts, r["censored"], FIGDIR / "durations.png")

    _report(alerts, r)


def _report(alerts: pd.DataFrame, r: dict) -> None:
    """Write `reports/eda.md` and echo it to stdout. Every line comes from `r`."""
    rate = r["rate"].set_index("year")["starts_per_day"]
    ratio = rate.loc[2026] / rate.loc[2022]
    peak_dow, peak_hour = r["peak"]
    gap = r["gaps"]
    b = r["burden"]
    brk = r["break"]
    cens = r["censored"]
    top5 = b.head(5)
    bottom3 = b.tail(3)

    lines = [
        "# EDA — air-raid alerts (national + per oblast)",
        "",
        "All numbers below are produced by `python -m src.eda`; figures in `reports/figures/`.",
        "",
        "## Data",
        f"- Window: {alerts['started_at'].min():%Y-%m-%d} → {alerts['started_at'].max():%Y-%m-%d} (UTC), "
        f"{len(alerts):,} alert-starts, {alerts['region'].nunique()} regions.",
        f"- Censored: naive={cens['naive']:.2%}; duration unusable (naive or >24h)={cens['duration']:.2%}.",
        "",
        "## Volume / regime",
        "- Starts/day by year: "
        + ", ".join(f"{y} {v:.1f}" for y, v in rate.items())
        + ".",
        f"- 2026 vs 2022 volume ratio: **{ratio:.2f}×** (per-day, partial-year safe).",
        f"- Change point ({brk['date']:%Y-%m-%d}, 28-day window): mean "
        f"{brk['before']:.1f} → {brk['after']:.1f} starts/day "
        f"({'up' if brk['after'] >= brk['before'] else 'down'}ward shift).",
        "",
        "## Seasonality",
        f"- Peak local hour-of-week: **{WEEKDAYS[peak_dow]} {peak_hour:02d}:00** (Europe/Kyiv).",
        f"- Median gap between consecutive alerts: **{gap.median():.2f} h**; "
        f"p90 gap: **{gap.quantile(0.9):.2f} h** ({gap.max() / 24:.0f} days max).",
        "",
        "## Per-oblast burden (sorted by total known alert-minutes)",
        "- Top 5:",
        *[f"  - {reg}: {b.loc[reg, 'alert_minutes']:,.0f} min, {b.loc[reg, 'starts']:,} starts, "
          f"{b.loc[reg, 'starts_per_100d']:.1f} starts/100d" for reg in top5.index],
        "- Bottom 3:",
        *[f"  - {reg}: {b.loc[reg, 'alert_minutes']:,.0f} min, {b.loc[reg, 'starts']:,} starts, "
          f"{b.loc[reg, 'starts_per_100d']:.1f} starts/100d" for reg in bottom3.index],
        "",
        "## Clustering",
        f"- {((gap < 1).mean()):.0%} of gaps are <1 h; "
        f"{((gap > 24).mean()):.0%} are >24 h → repeated starts inside mass attacks plus long quiet tails.",
        f"- Longest current quiet stretch: {r['since_days'].max():.1f} days.",
        "",
        "## Figures",
        "- `daily_starts.png`, `hour_of_week_heatmap.png`, `monthly_burden.png`, "
        "`gaps_and_quiet.png`, `region_burden.png`, `durations.png`",
    ]
    text = "\n".join(lines) + "\n"
    REPORT.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
