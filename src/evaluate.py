"""Honest holdout evaluation for the air-raid alert forecasts.

Order of operations — every choice is frozen on train/tune/folds before `test` is read:

    usable_origins -> split_days -> rolling_folds
    build train + tune + fold frames        (no test origin is built)
    choose_hyperparameters(folds)           (model selection only)
    fit on the 180-day train window -> predict tune
    choose_threshold + conformal_width(tune)
    build test frames                       <-- FIRST and ONLY read of `test`
    fit on train -> predict test -> metrics -> figures
                                          -> reports/metrics.json + reports/predictions.csv

The single test read is the `build_frames(splits["test"])` call in `main`; everything above
it uses `train`, `tune` or the fold windows, so no threshold/width/hyperparameter can have
seen a test label.

ponytail: 180-day rolling train window for every model and baseline so all numbers are
comparable and every fit stays in seconds (the series is non-stationary, 2026 volume is
2.28x 2022); upgrade path: expanding window with time-decay sample weights.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import pathlib
from concurrent.futures import ProcessPoolExecutor

import matplotlib

matplotlib.use("Agg")  # non-interactive: must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.features import build, load_alerts, load_grid, load_weather, origins
from src.models import (
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

FIGDIR = pathlib.Path("reports/figures")
METRICS = pathlib.Path("reports/metrics.json")
PREDICTIONS = pathlib.Path("reports/predictions.csv")

TRAIN_DAYS = 180
TEST_DAYS = 14
TUNE_DAYS = 14
N_FOLDS = 3
FOLD_DAYS = 14
TOLERANCES = (15, 30)
NOMINAL = 0.8
THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)

POISSON_ALPHAS = (0.1, 1.0, 10.0)
LOGISTIC_CS = (0.1, 1.0, 10.0)
HISTGB_GRID = ((0.05, 100), (0.05, 300), (0.1, 100), (0.1, 300))

HOUR = pd.Timedelta("24h")
WEEK = pd.Timedelta("7D")


# --------------------------------------------------------------------------- splits


def usable_origins(alerts: pd.DataFrame) -> list[pd.Timestamp]:
    """Daily origins whose whole 24 h horizon is observed: `origin + 24h <= max(finished_at)`.

    The last alerts end ~2026-10-11 04:46 UTC, so the last usable origin is Kyiv 2026-10-10.
    """
    last = alerts["finished_at"].max()
    first = alerts["started_at"].min()
    # The first day's horizon begins before the record does, so its labels would undercount.
    return [o for o in origins(first, last) if first <= o and o + HOUR <= last]


def _between(all_origins: list[pd.Timestamp], start, end) -> list[pd.Timestamp]:
    return [o for o in all_origins if start <= o < end]


def train_window(all_origins: list[pd.Timestamp], end, days: int = TRAIN_DAYS) -> list[pd.Timestamp]:
    """The last `days` daily origins strictly before `end` (rolling window, same for all)."""
    return _between(all_origins, end - pd.Timedelta(days=days), end)


def split_days(alerts: pd.DataFrame, test_days: int = TEST_DAYS, tune_days: int = TUNE_DAYS) -> dict:
    """Time-ordered, disjoint, non-overlapping windows: train < tune < test.

    `test` is the last `test_days` usable origins, `tune` the `tune_days` before it, `train`
    everything earlier. Nothing is shuffled.
    """
    usable = usable_origins(alerts)
    test = usable[-test_days:]
    tune = usable[-test_days - tune_days : -test_days]
    train = usable[: -test_days - tune_days]
    return {
        "train": train,
        "tune": tune,
        "test": test,
        "train_end": train[-1],
        "tune_start": tune[0],
        "test_start": test[0],
    }


def rolling_folds(
    alerts: pd.DataFrame, n_folds: int = N_FOLDS, fold_days: int = FOLD_DAYS, train_days: int = TRAIN_DAYS
) -> list[dict]:
    """Rolling-origin backtest for hyperparameter choice only.

    Fold 0 predicts the 14 days immediately before the tune window; each later fold steps
    back another 14 days. Every fold trains on the `train_days` origins before its eval
    window. Because the newest fold ends at `tune_start`, neither `tune` nor `test` can
    appear in any fold.
    """
    usable = usable_origins(alerts)
    boundary = split_days(alerts)["tune_start"]
    folds = []
    for k in range(n_folds):
        end = boundary - pd.Timedelta(days=fold_days * k)
        start = end - pd.Timedelta(days=fold_days)
        folds.append(
            {"fold": k, "train": train_window(usable, start, train_days), "eval": _between(usable, start, end)}
        )
    return folds


# --------------------------------------------------------------------------- frames


def _build_one(origin: pd.Timestamp) -> pd.DataFrame:
    return build(origin)


def build_frames(days: list[pd.Timestamp], workers: int = 4) -> pd.DataFrame:
    """Concatenated `build()` frames for `days`, sorted by origin (deterministic).

    One `build` call is ~1 s, so the ~250 origins a 3-fold backtest needs are built in
    parallel over forks; the parent warms `load_alerts/load_grid/load_weather` first so the
    forked workers inherit the caches instead of reloading the grid.
    """
    days = sorted(days)
    if not days:
        return pd.DataFrame()
    if workers > 1 and len(days) > 1:
        ctx = mp.get_context("fork")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            frames = list(ex.map(_build_one, days, chunksize=2))
    else:
        frames = [_build_one(o) for o in days]
    return pd.concat(frames, ignore_index=True)


def frame_for(big: pd.DataFrame, days: list[pd.Timestamp]) -> pd.DataFrame:
    """Rows of `big` whose origin is in `days`, sorted for a deterministic join order."""
    if not days:
        return big.iloc[:0]
    return (
        big[big["origin"].isin(pd.DatetimeIndex(sorted(days)))]
        .sort_values(["origin", "region", "hour"])
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------- baselines / metrics


def hour_of_week_daily(profile: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """Baseline A collapsed to an expected starts/day per origin x region (sum over 24 h)."""
    out = frame[["origin", "region"]].copy()
    out["pred"] = predict_hour_of_week(profile, frame)
    return out.groupby(["origin", "region"], as_index=False)["pred"].sum()


def seasonal_naive_daily(history: pd.DataFrame, frame: pd.DataFrame, target: str) -> pd.DataFrame:
    """Baseline B summed over the day: `target` at hour - 7d per region (0.0 when unseen).

    `history` is every observed hour up to the eval window (train + tune, and the eval
    window's own earlier days). A naive baseline has no parameters, and its lookup
    `hour - 7d` is always strictly before the row's origin, so reading it is leak-free.
    Mirrors `models.seasonal_naive`, generalised to `y_minutes`.
    """
    lookup = history.groupby(["region", "hour"])[target].mean()
    key = pd.MultiIndex.from_frame(frame[["region", "hour"]].assign(hour=frame["hour"] - WEEK))
    out = frame[["origin", "region"]].copy()
    out["pred"] = lookup.reindex(key).fillna(0.0).to_numpy(np.float64)
    return out.groupby(["origin", "region"], as_index=False)["pred"].sum()


def mae(pred, actual) -> float:
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(actual))))


def rmse(pred, actual) -> float:
    d = np.asarray(pred) - np.asarray(actual)
    return float(np.sqrt(np.mean(d * d)))


def count_metrics(panel: pd.DataFrame, pred_col: str, actual_col: str) -> dict:
    """Per-region-day MAE/RMSE plus national (sum over regions) MAE per origin."""
    nat = panel.groupby("origin")[[pred_col, actual_col]].sum()
    return {
        "mae": mae(panel[pred_col], panel[actual_col]),
        "rmse": rmse(panel[pred_col], panel[actual_col]),
        "national_mae": mae(nat[pred_col], nat[actual_col]),
    }


def match_events(pred_times, actual_times, tol_min: int) -> tuple[int, int, int]:
    """Point-wise event match within +/- `tol_min`: returns (tp, fp, fn).

    tp = actual events with a prediction within tolerance; fp = predictions with no actual
    nearby; fn = actual events missed. Predictions sit on the hour (:00), so a start at
    h:20 is 20 min from the h:00 prediction — inside 30, outside 15.
    """
    p = np.sort(pd.DatetimeIndex(pred_times).as_unit("ns").asi8)  # ns: pandas 3 asi8 is not always ns
    a = pd.DatetimeIndex(actual_times).as_unit("ns").asi8
    tol = int(pd.Timedelta(minutes=tol_min).value)
    if len(a) == 0:
        return 0, int(len(p)), 0
    if len(p) == 0:
        return 0, 0, int(len(a))

    def within(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
        i = np.searchsorted(src, tgt)
        left = np.clip(i - 1, 0, len(src) - 1)
        right = np.clip(i, 0, len(src) - 1)
        return np.minimum(np.abs(src[left] - tgt), np.abs(src[right] - tgt)) <= tol

    tp = int(within(p, a).sum())
    return tp, int((~within(a, p)).sum()), int(len(a) - tp)


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def actual_event_times(alerts: pd.DataFrame, start, end) -> dict:
    """Actual alert starts per region in `[start, end)`, at minute precision."""
    a = alerts[(alerts["started_at"] >= start) & (alerts["started_at"] < end)]
    return {region: pd.DatetimeIndex(g["started_at"]) for region, g in a.groupby("region")}


def _prob_groups(frame: pd.DataFrame, prob) -> list[tuple]:
    prob = np.asarray(prob)
    f = frame.reset_index(drop=True)
    return [(g["hour"].to_numpy(), prob[g.index.to_numpy()], region) for region, g in f.groupby("region")]


def event_counts(frame: pd.DataFrame, prob, events: dict, threshold: float, tol_min: int) -> tuple[int, int, int]:
    tp = fp = fn = 0
    for hours, p, region in _prob_groups(frame, prob):
        a, b, c = match_events(hours[p >= threshold], events.get(region, pd.DatetimeIndex([])), tol_min)
        tp, fp, fn = tp + a, fp + b, fn + c
    return tp, fp, fn


def best_threshold(frame: pd.DataFrame, prob, events: dict, tol_min: int = 30, grid=THRESHOLD_GRID) -> float:
    """Threshold maximising F1 at +/- `tol_min` on the given window (used on tune only)."""
    groups = _prob_groups(frame, prob)
    best_t, best_f1 = float(grid[0]), -1.0
    for t in grid:
        tp = fp = fn = 0
        for hours, p, region in groups:
            a, b, c = match_events(hours[p >= t], events.get(region, pd.DatetimeIndex([])), tol_min)
            tp, fp, fn = tp + a, fp + b, fn + c
        f1 = prf(tp, fp, fn)[2]
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t


def brier(prob, actual) -> float:
    return float(np.mean((np.asarray(prob) - np.asarray(actual)) ** 2))


def event_metrics(frame: pd.DataFrame, prob, events: dict, threshold: float) -> dict:
    out = {"threshold": float(threshold), "brier": brier(prob, frame["y_start"].to_numpy())}
    out["hourly_mae"] = float(np.mean(np.abs(np.asarray(prob) - frame["y_start"].to_numpy())))
    for tol in TOLERANCES:
        tp, fp, fn = event_counts(frame, prob, events, threshold, tol)
        p, r, f = prf(tp, fp, fn)
        out[f"tol_{tol}"] = {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f}
    return out


def conformal_width(pred, actual, level: float = NOMINAL) -> float:
    """Split-conformal half-width: `quantile(|residual|, level)` on the calibration window."""
    return float(np.quantile(np.abs(np.asarray(actual) - np.asarray(pred)), level))


def coverage(pred, actual, width: float) -> float:
    pred, actual = np.asarray(pred), np.asarray(actual)
    return float(np.mean((actual >= pred - width) & (actual <= pred + width)))


def duration_metrics(rows: pd.DataFrame, pred_col: str, width: float) -> dict:
    pred, actual = rows[pred_col].to_numpy(), rows["y_dur_log"].to_numpy()
    return {
        "n": int(len(rows)),
        "mae_log": float(np.mean(np.abs(pred - actual))),
        "mae_minutes": float(np.mean(np.abs(np.expm1(pred) - np.expm1(actual)))),
        "width": float(width),
        "coverage": coverage(pred, actual, width),
    }


def calibration_bins(prob, actual, bins: int = 10) -> list[dict]:
    """Reliability curve: predicted mean vs observed frequency per equal-width probability bin."""
    prob, actual = np.asarray(prob), np.asarray(actual)
    idx = np.clip(np.digitize(prob, np.linspace(0, 1, bins + 1)[1:-1]), 0, bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        if m.any():
            out.append({"bin": b, "n": int(m.sum()), "pred": float(prob[m].mean()), "obs": float(actual[m].mean())})
    return out


def region_share(frame: pd.DataFrame, prob) -> list[dict]:
    """Per-oblast observed vs predicted share of hours with an alert start."""
    f = frame[["region"]].copy()
    f["pred"] = np.asarray(prob)
    f["obs"] = frame["y_start"].to_numpy()
    g = f.groupby("region", as_index=False).mean().sort_values("region")
    return [{"region": r, "n": int(n), "pred": p, "obs": o} for r, n, p, o in
            zip(g["region"], frame.groupby("region").size().reindex(g["region"]), g["pred"], g["obs"])]


# --------------------------------------------------------------------------- predictions


def daily_predictions(train_frame: pd.DataFrame, eval_frame: pd.DataFrame, hp: dict, history: pd.DataFrame) -> pd.DataFrame:
    tr, ev = daily_panel(train_frame), daily_panel(eval_frame).copy()
    profile = hour_of_week_profile(train_frame)
    for name, d in (
        ("p_how", hour_of_week_daily(profile, eval_frame)),
        ("p_snaive_start", seasonal_naive_daily(history, eval_frame, "y_start")),
        ("p_snaive_min", seasonal_naive_daily(history, eval_frame, "y_minutes")),
    ):
        ev = ev.merge(d.rename(columns={"pred": name}), on=["origin", "region"], how="left")
    ev["p_poisson_start"] = fit_predict(make_poisson(alpha=hp["poisson_alpha"]), tr, ev, "y_starts_day")
    ev["p_poisson_min"] = fit_predict(make_poisson(alpha=hp["poisson_alpha"]), tr, ev, "y_minutes_day")
    ev["p_histgb_start"] = fit_predict(make_histgb(loss="poisson", **hp["histgb_poisson"]), tr, ev, "y_starts_day")
    ev["p_histgb_min"] = fit_predict(make_histgb(loss="poisson", **hp["histgb_poisson"]), tr, ev, "y_minutes_day")
    return ev


def hourly_predictions(train_frame: pd.DataFrame, eval_frame: pd.DataFrame, hp: dict, history: pd.DataFrame) -> pd.DataFrame:
    ev = eval_frame.copy().reset_index(drop=True)
    profile = hour_of_week_profile(train_frame)
    ev["p_how"] = predict_hour_of_week(profile, eval_frame)
    ev["p_snaive"] = seasonal_naive(history, eval_frame)
    ev["p_logistic"] = fit_predict(make_logistic(C=hp["logistic_C"]), train_frame, eval_frame, "y_start")
    histgb = fit_predict(make_histgb(loss="poisson", **hp["histgb_poisson"]), train_frame, eval_frame, "y_start")
    ev["p_histgb"] = np.clip(histgb, 0.0, 1.0)  # a Poisson rate read as a probability
    return ev


def duration_predictions(train_frame: pd.DataFrame, eval_frame: pd.DataFrame, hp: dict) -> pd.DataFrame:
    ev = duration_rows(eval_frame).copy().reset_index(drop=True)
    ev["p_naive"] = duration_naive(train_frame, ev)
    ev["p_histgb"] = fit_predict(
        make_histgb(loss="quantile", quantile=0.5, **hp["histgb_duration"]), duration_rows(train_frame), ev, "y_dur_log"
    )
    return ev


# --------------------------------------------------------------------------- hyperparameter choice


def choose_hyperparameters(folds: list[dict], big: pd.DataFrame, alerts: pd.DataFrame) -> tuple[dict, list[dict]]:
    """Pick each learner's config by mean fold score (MAE for counts/duration, F1 for events)."""
    poisson = {a: [] for a in POISSON_ALPHAS}
    logistic = {c: [] for c in LOGISTIC_CS}
    histgb = {g: [] for g in HISTGB_GRID}
    duration = {g: [] for g in HISTGB_GRID}
    table = []

    for fold in folds:
        tr, ev = frame_for(big, fold["train"]), frame_for(big, fold["eval"])
        trp, evp = daily_panel(tr), daily_panel(ev)
        row = {"fold": fold["fold"], "poisson_alpha_mae": {}, "logistic_C_f1": {},
               "histgb_poisson_mae": {}, "histgb_duration_mae": {}}
        for a in POISSON_ALPHAS:
            row["poisson_alpha_mae"][str(a)] = mae(
                fit_predict(make_poisson(alpha=a), trp, evp, "y_starts_day"), evp["y_starts_day"]
            )
            poisson[a].append(row["poisson_alpha_mae"][str(a)])
        for g in HISTGB_GRID:
            lr, mi = g
            m = make_histgb(loss="poisson", learning_rate=lr, max_iter=mi)
            row["histgb_poisson_mae"][f"{lr}_{mi}"] = mae(
                fit_predict(m, trp, evp, "y_starts_day"), evp["y_starts_day"]
            )
            histgb[g].append(row["histgb_poisson_mae"][f"{lr}_{mi}"])
        trd, evd = duration_rows(tr), duration_rows(ev)
        for g in HISTGB_GRID:
            lr, mi = g
            m = make_histgb(loss="quantile", quantile=0.5, learning_rate=lr, max_iter=mi)
            row["histgb_duration_mae"][f"{lr}_{mi}"] = mae(fit_predict(m, trd, evd, "y_dur_log"), evd["y_dur_log"])
            duration[g].append(row["histgb_duration_mae"][f"{lr}_{mi}"])
        events = actual_event_times(alerts, ev["hour"].min(), ev["hour"].max() + pd.Timedelta("1h"))
        for c in LOGISTIC_CS:
            prob = fit_predict(make_logistic(C=c), tr, ev, "y_start")
            t = best_threshold(ev, prob, events)
            row["logistic_C_f1"][str(c)] = prf(*event_counts(ev, prob, events, t, 30))[2]
            logistic[c].append(row["logistic_C_f1"][str(c)])
        table.append(row)

    def pick(scores, best):
        return best(scores, key=lambda k: float(np.mean(scores[k])))

    alpha = pick(poisson, min)
    c = pick(logistic, max)
    g_count = pick(histgb, min)
    g_dur = pick(duration, min)
    hp = {
        "poisson_alpha": float(alpha),
        "logistic_C": float(c),
        "histgb_poisson": {"learning_rate": g_count[0], "max_iter": g_count[1]},
        "histgb_duration": {"learning_rate": g_dur[0], "max_iter": g_dur[1]},
    }
    return hp, table


def fold_mean(table: list[dict]) -> dict:
    """Mean over folds of every candidate score, so the choice is auditable in metrics.json."""
    out = {}
    for key in ("poisson_alpha_mae", "logistic_C_f1", "histgb_poisson_mae", "histgb_duration_mae"):
        names = table[0][key].keys()
        out[key] = {n: float(np.mean([row[key][n] for row in table])) for n in names}
    return out


# --------------------------------------------------------------------------- figures


def _fig_national(panel: pd.DataFrame, path: pathlib.Path) -> None:
    nat = panel.groupby("origin")[["y_starts_day", "p_how", "p_snaive_start", "p_poisson_start", "p_histgb_start"]].sum()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(nat.index, nat["y_starts_day"], "k.-", lw=2, label="actual")
    for col, lab in (("p_how", "A hour-of-week"), ("p_snaive_start", "B seasonal naive"),
                     ("p_poisson_start", "Poisson"), ("p_histgb_start", "HistGB")):
        ax.plot(nat.index, nat[col], marker="o", ms=3, lw=1, label=lab)
    ax.set(title="National daily alert starts — holdout (test)", ylabel="starts / day")
    ax.legend(ncol=3, fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _fig_calibration(curves: dict, path: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for name, rows in curves.items():
        if rows:
            ax.plot([r["pred"] for r in rows], [r["obs"] for r in rows], "o-", label=name)
    ax.set(xlabel="mean predicted P(start)", ylabel="observed start frequency",
           title="Start-probability calibration — holdout")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _fig_region_share(shares: dict, path: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6))
    hi = max(max([r["obs"] for r in rows] + [r["pred"] for r in rows]) for rows in shares.values()) * 1.05
    ax.plot([0, hi], [0, hi], "k--", lw=1, label="perfect")
    for name, rows in shares.items():
        ax.scatter([r["obs"] for r in rows], [r["pred"] for r in rows], s=18, alpha=0.7, label=name)
    ax.set(xlabel="observed alert-time share", ylabel="predicted alert-time share",
           title="Per-oblast observed vs predicted alert-time share — holdout")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _fig_duration(rows: pd.DataFrame, path: pathlib.Path) -> None:
    s = rows.sort_values("p_histgb").reset_index(drop=True)
    x = np.arange(len(s))
    actual, pred, width = s["y_dur_log"].to_numpy(), s["p_histgb"].to_numpy(), s.attrs["width"]
    cov = s.attrs["coverage"]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.errorbar(x, pred, yerr=width, fmt="none", ecolor="tab:orange", alpha=0.5, label=f"80% conformal (w={width:.2f})")
    ax.plot(x, pred, color="tab:orange", lw=1, label="HistGB q50")
    ax.plot(x, actual, ".", color="k", ms=3, label="actual log-duration")
    ax.set(xlabel="test alert-starts (sorted by prediction)", ylabel="log1p(minutes)",
           title=f"Duration conformal intervals — empirical coverage {cov:.1%} (nominal {NOMINAL:.0%})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _fig_folds(table: list[dict], path: pathlib.Path) -> None:
    folds = [r["fold"] for r in table]
    x = np.arange(len(folds))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for key, lab in (("poisson_alpha_mae", "Poisson"), ("histgb_poisson_mae", "HistGB counts"),
                     ("histgb_duration_mae", "HistGB duration")):
        ax1.plot(x, [min(r[key].values()) for r in table], "o-", label=lab)
    ax1.set(xticks=x, xticklabels=[f"fold {f}" for f in folds], ylabel="best MAE in fold",
            title="Rolling-origin fold MAE (lower is better)")
    ax1.legend(fontsize=8)
    ax2.plot(x, [max(r["logistic_C_f1"].values()) for r in table], "s-", color="tab:red")
    ax2.set(xticks=x, xticklabels=[f"fold {f}" for f in folds], ylabel="best F1 in fold",
            title="Rolling-origin fold F1 (start events, +/-30 min)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- driver


def _range(days: list[pd.Timestamp]) -> list:
    return [days[0].isoformat(), days[-1].isoformat(), len(days)] if days else None


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    alerts = load_alerts()
    load_grid()
    load_weather()  # warm caches so forked build workers inherit them

    usable = usable_origins(alerts)
    splits = split_days(alerts)
    folds = rolling_folds(alerts)

    need = set(splits["tune"]) | set(train_window(usable, splits["tune_start"]))
    for f in folds:
        need |= set(f["train"]) | set(f["eval"])
    big = build_frames(sorted(need))  # no test origin here

    hp, fold_table = choose_hyperparameters(folds, big, alerts)
    train_frame = frame_for(big, train_window(usable, splits["tune_start"]))
    tune_frame = frame_for(big, splits["tune"])

    # --- freeze threshold + conformal width on the TUNE window ---
    tune_h = hourly_predictions(train_frame, tune_frame, hp, history=big)
    tune_events = actual_event_times(alerts, tune_frame["hour"].min(), tune_frame["hour"].max() + pd.Timedelta("1h"))
    hourly_models = {"hour_of_week": "p_how", "seasonal_naive": "p_snaive",
                     "logistic": "p_logistic", "histgb": "p_histgb"}
    thresholds = {name: best_threshold(tune_h, tune_h[col], tune_events) for name, col in hourly_models.items()}

    tune_d = duration_predictions(train_frame, tune_frame, hp)
    widths = {name: conformal_width(tune_d[col], tune_d["y_dur_log"])
              for name, col in (("duration_naive", "p_naive"), ("histgb_q50", "p_histgb"))}

    # ==================== FIRST AND ONLY READ OF `test` ====================
    test_frame = build_frames(splits["test"])
    history = pd.concat([big, test_frame], ignore_index=True)  # naive lookback needs tune + test's own past

    test_daily = daily_predictions(train_frame, test_frame, hp, history=history)
    test_h = hourly_predictions(train_frame, test_frame, hp, history=history)
    test_events = actual_event_times(alerts, test_frame["hour"].min(), test_frame["hour"].max() + pd.Timedelta("1h"))
    test_dur = duration_predictions(train_frame, test_frame, hp)

    counts = {}
    for target, preds in (
        ("y_starts_day", {"hour_of_week": "p_how", "seasonal_naive": "p_snaive_start",
                          "poisson": "p_poisson_start", "histgb": "p_histgb_start"}),
        ("y_minutes_day", {"seasonal_naive": "p_snaive_min", "poisson": "p_poisson_min", "histgb": "p_histgb_min"}),
    ):
        counts[target] = {name: count_metrics(test_daily, col, target) for name, col in preds.items()}

    events = {name: event_metrics(test_h, test_h[col], test_events, thresholds[name])
              for name, col in hourly_models.items()}

    duration = {name: duration_metrics(test_dur, col, widths[name])
                for name, col in (("duration_naive", "p_naive"), ("histgb_q50", "p_histgb"))}

    curves = {name: calibration_bins(test_h[col], test_h["y_start"]) for name, col in
              (("logistic", "p_logistic"), ("histgb", "p_histgb"))}
    shares = {name: region_share(test_h, test_h[col]) for name, col in
              (("logistic", "p_logistic"), ("histgb", "p_histgb"))}

    # every model vs every baseline present, so "cannot beat the usual times" is explicit
    for target in counts:
        baselines = [b for b in counts[target] if b in ("hour_of_week", "seasonal_naive")]
        for name, m in counts[target].items():
            m["beats_baselines"] = {
                b: bool(name != b and m["mae"] < counts[target][b]["mae"]) for b in baselines
            }

    result = {
        "data": {
            "started_at_min": alerts["started_at"].min().isoformat(),
            "finished_at_max": alerts["finished_at"].max().isoformat(),
            "rows": int(len(alerts)),
            "regions": int(alerts["region"].nunique()),
        },
        "setup": {
            "train_days": TRAIN_DAYS, "test_days": TEST_DAYS, "tune_days": TUNE_DAYS,
            "n_folds": N_FOLDS, "fold_days": FOLD_DAYS, "nominal_coverage": NOMINAL,
        },
        "splits": {
            "usable_origins": len(usable),
            "train": _range(splits["train"]), "tune": _range(splits["tune"]), "test": _range(splits["test"]),
            "holdout_train_window": _range(train_window(usable, splits["tune_start"])),
        },
        "folds": [
            {"fold": f["fold"], "train": _range(f["train"]), "eval": _range(f["eval"])} for f in folds
        ],
        "fold_scores": fold_table,
        "fold_mean_scores": fold_mean(fold_table),
        "hyperparameters": hp,
        "thresholds": thresholds,
        "conformal_widths": widths,
        "counts_daily": counts,
        "start_events": events,
        "duration": duration,
        "calibration": {"bins": curves, "per_region": shares},
    }

    _fig_national(test_daily, FIGDIR / "eval_national_daily.png")
    _fig_calibration(curves, FIGDIR / "eval_calibration.png")
    _fig_region_share(shares, FIGDIR / "eval_region_share.png")
    test_dur.attrs["width"] = widths["histgb_q50"]
    test_dur.attrs["coverage"] = duration["histgb_q50"]["coverage"]
    _fig_duration(test_dur, FIGDIR / "eval_duration_conformal.png")
    _fig_folds(fold_table, FIGDIR / "eval_fold_scores.png")

    METRICS.write_text(json.dumps(result, indent=2, default=_jsonable) + "\n")
    # The raw holdout guesses, one row per origin x region: src.brief turns these into a
    # plain-language "what really happened vs what we said" table for non-technical readers.
    panel_cols = ["origin", "region", "y_starts_day", "y_minutes_day", "p_how",
                  "p_snaive_start", "p_snaive_min", "p_poisson_start", "p_poisson_min",
                  "p_histgb_start", "p_histgb_min"]
    test_daily[panel_cols].to_csv(PREDICTIONS, index=False)
    _summary(result)


def _jsonable(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"not JSON-serializable: {type(o)}")


def _summary(r: dict) -> None:
    print("=" * 78)
    print("ALERT FORECAST — HONEST HOLDOUT EVALUATION")
    print("=" * 78)
    print(f"data: {r['data']['started_at_min'][:10]} -> {r['data']['finished_at_max'][:16]} UTC, "
          f"{r['data']['rows']:,} alerts, {r['data']['regions']} regions")
    for name, s in (("train", r["splits"]["train"]), ("tune", r["splits"]["tune"]), ("test", r["splits"]["test"])):
        print(f"  {name:5s}: {s[0][:10]} -> {s[1][:10]}  ({s[2]} days)")
    print(f"  holdout train window (last {r['setup']['train_days']}d before tune): "
          f"{r['splits']['holdout_train_window'][0][:10]} -> {r['splits']['holdout_train_window'][1][:10]}")

    print("\nfolds (rolling-origin, hyperparameter choice only):")
    for f, m in zip(r["folds"], r["fold_scores"]):
        print(f"  fold {f['fold']}: train {f['train'][0][:10]}->{f['train'][1][:10]}, "
              f"eval {f['eval'][0][:10]}->{f['eval'][1][:10]} | "
              f"poisson MAE {min(m['poisson_alpha_mae'].values()):.3f}, "
              f"histgb MAE {min(m['histgb_poisson_mae'].values()):.3f}, "
              f"logistic F1 {max(m['logistic_C_f1'].values()):.3f}")
    print(f"  chosen: poisson alpha={r['hyperparameters']['poisson_alpha']}, "
          f"logistic C={r['hyperparameters']['logistic_C']}, "
          f"histgb poisson={r['hyperparameters']['histgb_poisson']}, "
          f"histgb duration={r['hyperparameters']['histgb_duration']}")

    print("\ncounts/day — starts (MAE / RMSE / national MAE):")
    for name, m in r["counts_daily"]["y_starts_day"].items():
        beats = ", ".join(f"{b}:{v}" for b, v in m["beats_baselines"].items())
        print(f"  {name:14s} {m['mae']:6.3f}  {m['rmse']:6.3f}  {m['national_mae']:7.3f}   beats[{beats}]")
    print("counts/day — minutes (MAE / RMSE / national MAE):")
    for name, m in r["counts_daily"]["y_minutes_day"].items():
        beats = ", ".join(f"{b}:{v}" for b, v in m["beats_baselines"].items())
        print(f"  {name:14s} {m['mae']:7.2f} {m['rmse']:7.2f} {m['national_mae']:8.2f}   beats[{beats}]")

    print("\nstart events (threshold from tune, F1 @ +/-30 min):")
    for name, m in r["start_events"].items():
        print(f"  {name:14s} thr={m['threshold']:.2f}  F1@15={m['tol_15']['f1']:.3f}  "
              f"F1@30={m['tol_30']['f1']:.3f}  P={m['tol_30']['precision']:.3f} "
              f"R={m['tol_30']['recall']:.3f}  Brier={m['brier']:.3f}")

    print("\nduration (log MAE / minutes MAE / conformal width / coverage):")
    for name, m in r["duration"].items():
        print(f"  {name:14s} logMAE={m['mae_log']:.3f}  minMAE={m['mae_minutes']:.1f}  "
              f"width={m['width']:.3f}  coverage={m['coverage']:.1%} (nominal {NOMINAL:.0%})")
    print(f"\nwrote {METRICS} and eval_* figures in {FIGDIR}")


if __name__ == "__main__":
    main()
