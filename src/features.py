"""
Gap-safe causal priors, nocturnal boundary-layer features, and Model 0 heuristic baseline.

Design logic:
- ZERO row-index `.shift()` operations: Mumbai telemetry contains a 5-day withheld date
  hole (Oct 24-28), so all lookbacks use O(log N) calendar slicing via `np.searchsorted`
  bounded strictly to (cutoff_ts - 35 days, cutoff_ts].
- Multi-day lags are computed on a clipped series (`clip_cap = 190.0`) to prevent Diwali
  firework peaks from inflating Day +3/+4 sea-breeze forecasts.
- Blends median with clipped mean (`mean_blend = 0.45`) to correct median under-prediction
  bias on right-skewed winter PM2.5 distributions under RMSE.
- Supports optional `trace: dict | None = None` for glass-box inspection in VS Code.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.config import FEATURE_COLS, MAX_CORES, TIMEZONE_IST


def compute_cutoff_priors(
    df_hist: pd.DataFrame,
    cutoff_ts: pd.Timestamp,
    clip_cap: float = 190.0,
    mean_blend: float = 0.45,
    trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Computes strictly causal station x hour diurnal anchors and 72h/24h/6h anomalies
    using observations strictly <= cutoff_ts within a 35-day lookback window.
    """
    ts_series = df_hist["timestamp_ist"]
    low_ts = cutoff_ts - pd.Timedelta(days=35)
    idx_start = int(ts_series.searchsorted(low_ts, side="right"))
    idx_end = int(ts_series.searchsorted(cutoff_ts, side="right"))
    valid = df_hist.iloc[idx_start:idx_end].copy()

    valid["hour_int"] = valid["timestamp_ist"].dt.hour
    valid["pm25_clip"] = valid["pm25"].clip(upper=clip_cap)

    def _blended(med_s: pd.Series, mean_s: pd.Series) -> pd.Series:
        return (1.0 - mean_blend) * med_s + mean_blend * mean_s

    # 1. 14-day station x hour anchor (calendar 14d if >= 8 obs, else last 14 active obs)
    w14_cal = valid[valid["timestamp_ist"] > (cutoff_ts - pd.Timedelta(days=14))]
    c14 = w14_cal.groupby(["location_id", "hour_int"])["pm25_clip"].agg(["median", "mean", "count"])
    a14 = valid.groupby(["location_id", "hour_int"]).tail(14).groupby(["location_id", "hour_int"])["pm25_clip"].agg(
        med_a="median", mean_a="mean"
    )
    m14 = pd.concat([c14, a14], axis=1)
    use_cal = m14["count"] >= 8
    d14_prior = pd.Series(
        np.where(use_cal, _blended(m14["median"], m14["mean"]), _blended(m14["med_a"], m14["mean_a"])),
        index=m14.index,
        name="diurnal_14d_prior",
    )
    d14_med = pd.Series(np.where(use_cal, m14["median"], m14["med_a"]), index=m14.index, name="diurnal_14d_med")

    # 2. Recent 5-active-obs, 7-day, and 3-day station x hour priors on clipped series
    a5 = valid.groupby(["location_id", "hour_int"]).tail(5).groupby(["location_id", "hour_int"])["pm25_clip"].agg(["median", "mean"])
    d5_act = pd.Series(_blended(a5["median"], a5["mean"]), index=a5.index, name="diurnal_5d_act")

    w7 = valid[valid["timestamp_ist"] > (cutoff_ts - pd.Timedelta(days=7))]
    c7 = w7.groupby(["location_id", "hour_int"])["pm25_clip"].agg(["median", "mean"])
    d7 = pd.Series(_blended(c7["median"], c7["mean"]), index=c7.index, name="diurnal_7d_prior")

    w3 = valid[valid["timestamp_ist"] > (cutoff_ts - pd.Timedelta(days=3))]
    c3 = w3.groupby(["location_id", "hour_int"])["pm25_clip"].agg(["median", "mean"])
    d3 = pd.Series(_blended(c3["median"], c3["mean"]), index=c3.index, name="diurnal_3d_prior")

    p_df = pd.concat([d14_med, d14_prior, d7, d5_act, d3], axis=1)
    global_med = float(valid["pm25"].tail(14 * 24 * 4).median()) if len(valid) > 0 else 45.0
    p_df["diurnal_14d_med"] = p_df["diurnal_14d_med"].fillna(global_med)
    p_df["diurnal_14d_prior"] = p_df["diurnal_14d_prior"].fillna(p_df["diurnal_14d_med"])
    p_df["diurnal_5d_act"] = p_df["diurnal_5d_act"].fillna(p_df["diurnal_14d_prior"])
    p_df["diurnal_7d_prior"] = p_df["diurnal_7d_prior"].fillna(p_df["diurnal_5d_act"])
    p_df["diurnal_3d_prior"] = p_df["diurnal_3d_prior"].fillna(p_df["diurnal_5d_act"])

    # 3. Hour-matched regime anomalies over last 72h, 24h, and 6h
    w72h = valid[valid["timestamp_ist"] > (cutoff_ts - pd.Timedelta(hours=72))].copy()
    w72h = w72h.merge(p_df[["diurnal_14d_prior"]].reset_index(), on=["location_id", "hour_int"], how="left")
    w72h["hourly_anom"] = (w72h["pm25_clip"] - w72h["diurnal_14d_prior"]).clip(-45.0, 80.0)

    w24h = w72h[w72h["timestamp_ist"] > (cutoff_ts - pd.Timedelta(hours=24))]
    w6h = w72h[w72h["timestamp_ist"] > (cutoff_ts - pd.Timedelta(hours=6))]

    priors = {
        "priors_df": p_df.reset_index(),
        "st_6h_anom": w6h.groupby("location_id")["hourly_anom"].mean().rename("st_6h_anom"),
        "st_24h_anom": w24h.groupby("location_id")["hourly_anom"].mean().rename("st_24h_anom"),
        "st_72h_anom": w72h.groupby("location_id")["hourly_anom"].mean().rename("st_72h_anom"),
        "st_72h_std": w72h.groupby("location_id")["hourly_anom"].std().rename("station_72h_std"),
        "city_72h_anom": float(w72h["hourly_anom"].mean()) if len(w72h) > 0 else 0.0,
        "city_24h_anom": float(w24h["hourly_anom"].mean()) if len(w24h) > 0 else 0.0,
        "city_6h_anom": float(w6h["hourly_anom"].mean()) if len(w6h) > 0 else 0.0,
        "global_med": global_med,
    }
    if len(w24h) == 0:
        priors["city_24h_anom"] = priors["city_72h_anom"]
    if len(w6h) == 0:
        priors["city_6h_anom"] = priors["city_24h_anom"]

    if trace is not None:
        trace["priors_df"] = priors["priors_df"].copy()
        trace["w72h_anomalies"] = w72h.copy()

    return priors


def build_features_for_window(
    df_window: pd.DataFrame,
    cutoff_ts: pd.Timestamp,
    priors: dict[str, Any],
    time_col: str,
    prior_weights: tuple[float, float, float] = (0.45, 0.35, 0.20),
    tau_fast: float = 30.0,
    tau_slow: float = 120.0,
    trace: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Constructs horizon-decayed baseline_prior, two-speed exponential regime anomalies,
    regime velocity features, and nocturnal boundary-layer inversion interactions.
    """
    out = df_window.reset_index(drop=True).copy()
    ts = out[time_col]
    out["hour_int"] = ts.dt.hour
    out["day_of_week"] = ts.dt.day_of_week
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["cutoff_hour"] = int(cutoff_ts.hour)
    out["hour_sin"] = np.sin(2.0 * np.pi * out["hour_int"] / 24.0)
    out["hour_cos"] = np.cos(2.0 * np.pi * out["hour_int"] / 24.0)

    # Merge causal station x hour priors and lookback anomalies
    out = out.merge(priors["priors_df"], on=["location_id", "hour_int"], how="left")
    for key in ("st_6h_anom", "st_24h_anom", "st_72h_anom", "st_72h_std"):
        out = out.merge(priors[key].reset_index(), on="location_id", how="left")

    for col in ("diurnal_14d_med", "diurnal_14d_prior", "diurnal_7d_prior", "diurnal_5d_act", "diurnal_3d_prior"):
        out[col] = out[col].fillna(priors["global_med"])

    out["st_72h_anom"] = out["st_72h_anom"].fillna(0.0)
    out["st_24h_anom"] = out["st_24h_anom"].fillna(out["st_72h_anom"])
    out["st_6h_anom"] = out["st_6h_anom"].fillna(out["st_24h_anom"])
    out["station_72h_std"] = out["station_72h_std"].fillna(0.0)

    # Forecast lead time in hours (keep both raw and clipped horizons)
    raw_horizon = np.clip((out[time_col] - cutoff_ts).dt.total_seconds().to_numpy() / 3600.0, 1.0, None)
    out["raw_horizon_hours"] = raw_horizon
    out["horizon_hours"] = np.clip(raw_horizon, 1.0, 64.0)

    # Horizon-decayed adaptive baseline_prior (relaxes back to 14d anchor across multi-day holes)
    _, w_7, w_5 = prior_weights
    gamma_base = np.exp(-np.maximum(0.0, raw_horizon - 24.0) / 72.0)
    d14_p = out["diurnal_14d_prior"].to_numpy()
    d7_p = out["diurnal_7d_prior"].to_numpy()
    d5_p = out["diurnal_5d_act"].to_numpy()
    d3_p = out["diurnal_3d_prior"].to_numpy()
    out["baseline_prior"] = d14_p + gamma_base * (w_7 * (d7_p - d14_p) + w_5 * (d5_p - d14_p))

    # Two-speed exponential decay + nocturnal boundary-layer inversion weight (peaks at 04:00 IST)
    decay_fast = np.exp(-raw_horizon / tau_fast)
    decay_slow = np.exp(-raw_horizon / tau_slow)
    inv_weight = 0.5 * (1.0 + np.cos(2.0 * np.pi * (out["hour_int"].to_numpy() - 4.0) / 24.0))
    out["inversion_weight"] = inv_weight

    st_6h, st_24h, st_72h = out["st_6h_anom"].to_numpy(), out["st_24h_anom"].to_numpy(), out["st_72h_anom"].to_numpy()
    c_24h, c_72h = float(priors["city_24h_anom"]), float(priors["city_72h_anom"])

    out["slow_st_72h_anom"] = st_72h * decay_slow
    out["slow_city_72h_anom"] = c_72h * decay_slow
    out["slow_d5_vs_d14"] = (d5_p - d14_p).clip(-45.0, 65.0) * decay_slow
    out["slow_d7_vs_d14"] = (d7_p - d14_p).clip(-45.0, 65.0) * decay_slow

    out["fast_st_6h_anom"] = st_6h * decay_fast
    out["fast_st_24h_anom"] = st_24h * decay_fast
    out["fast_city_24h_anom"] = c_24h * decay_fast
    out["fast_d3_vs_base"] = (d3_p - out["baseline_prior"].to_numpy()).clip(-45.0, 65.0) * decay_fast

    out["vel_6h_vs_24h"] = (st_6h - st_24h) * decay_fast
    out["vel_24h_vs_72h"] = (st_24h - st_72h) * decay_fast
    out["city_vel_24h_vs_72h"] = (c_24h - c_72h) * decay_fast

    out["inv_x_st_72h"] = inv_weight * out["slow_st_72h_anom"]
    out["inv_x_st_24h"] = inv_weight * out["fast_st_24h_anom"]
    out["st_vs_city_72h"] = (st_72h - c_72h) * decay_slow

    if trace is not None:
        trace["window_features"] = out.copy()

    return out


def _build_single_cutoff_window(
    df_series: pd.DataFrame,
    c_ts: pd.Timestamp,
    is_2024_hist: bool,
    clip_cap: float,
    mean_blend: float,
) -> pd.DataFrame | None:
    """Builds one causal 64-hour supervised training window starting strictly after c_ts."""
    ts_col = df_series["timestamp_ist"]
    i_lb0 = int(ts_col.searchsorted(c_ts - pd.Timedelta(hours=72), side="right"))
    i_c = int(ts_col.searchsorted(c_ts, side="right"))
    i_tg1 = int(ts_col.searchsorted(c_ts + pd.Timedelta(hours=64), side="right"))

    lb_slice = df_series.iloc[i_lb0:i_c]
    tg_slice = df_series.iloc[i_c:i_tg1]
    if len(lb_slice) < 36 or len(tg_slice) < 36:
        return None

    lb_counts = lb_slice.groupby("location_id")["pm25"].count()
    tg_counts = tg_slice.groupby("location_id")["pm25"].count()
    valid_locs = [loc for loc in lb_counts.index if lb_counts.get(loc, 0) >= 36 and tg_counts.get(loc, 0) >= 36]
    if not valid_locs:
        return None

    tg_valid = tg_slice[tg_slice["location_id"].isin(valid_locs)].copy()
    priors = compute_cutoff_priors(df_series, c_ts, clip_cap=clip_cap, mean_blend=mean_blend)
    feat_df = build_features_for_window(tg_valid, c_ts, priors, time_col="timestamp_ist")
    feat_df["cutoff_ts"] = c_ts
    feat_df["is_2024_hist"] = int(is_2024_hist)
    feat_df["target_residual"] = feat_df["pm25"] - feat_df["baseline_prior"]
    feat_df["target_log_ratio"] = np.log(feat_df["pm25"] + 10.0) - np.log(feat_df["baseline_prior"] + 10.0)
    return feat_df


def build_supervised_dataset(
    df_series: pd.DataFrame,
    label: str = "2025_2026",
    is_2024_hist: bool = False,
    clip_cap: float = 190.0,
    mean_blend: float = 0.45,
    n_jobs: int = MAX_CORES,
) -> pd.DataFrame:
    """
    Generates all supervised 64h windows anchored at 09:00 IST and 21:00 IST in parallel
    across CPU threads. Enforces zero NaNs in FEATURE_COLS.
    """
    min_ts = df_series["timestamp_ist"].min() + pd.Timedelta(days=14)
    max_ts = df_series["timestamp_ist"].max() - pd.Timedelta(hours=40)
    if min_ts >= max_ts:
        return pd.DataFrame()

    base_days = pd.date_range(start=min_ts.floor("D"), end=max_ts.floor("D"), freq="1D", tz=TIMEZONE_IST)
    candidate_cutoffs = (base_days + pd.Timedelta(hours=9)).union(base_days + pd.Timedelta(hours=21)).sort_values()
    candidate_cutoffs = candidate_cutoffs[(candidate_cutoffs >= min_ts) & (candidate_cutoffs <= max_ts)]

    parts = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_build_single_cutoff_window)(df_series, c_ts, is_2024_hist, clip_cap, mean_blend)
        for c_ts in candidate_cutoffs
    )
    windows = [p for p in parts if p is not None]
    if not windows:
        return pd.DataFrame()

    sup_df = pd.concat(windows, ignore_index=True)
    sup_df["location_id"] = sup_df["location_id"].astype("category")

    total_nans = int(sup_df[FEATURE_COLS].isna().sum().sum())
    assert total_nans == 0, f"Found {total_nans} NaNs in FEATURE_COLS for {label}!"

    print(
        f"[Validation Check 3] Built '{label}' Supervised Set | Rows={len(sup_df):,} | "
        f"Cols={sup_df.shape[1]} | Cutoffs={len(windows):,} | MaxCutoff={sup_df['cutoff_ts'].max()} | NaNs=0"
    )
    return sup_df


def build_model0_heuristic(
    train_clean: pd.DataFrame,
    test_raw: pd.DataFrame,
    causal_cutoffs: list[pd.Timestamp],
) -> pd.DataFrame:
    """
    Zero-ML Model 0 Baseline: 14-day gap-safe station x hour median prior.
    Proves end-to-end ID alignment, timezone matching, and metric correctness before ML.
    """
    priors_map = {c_ts: compute_cutoff_priors(train_clean, c_ts) for c_ts in causal_cutoffs}
    test_work = test_raw[["ID", "location_id", "target_end_ist", "scoring_block"]].copy()
    test_work["assigned_cutoff"] = test_work["target_end_ist"].apply(
        lambda ts: [c for c in causal_cutoffs if c < ts][-1] if any(c < ts for c in causal_cutoffs) else causal_cutoffs[0]
    )

    parts = []
    for c_ts, grp in test_work.groupby("assigned_cutoff", sort=False):
        f_grp = build_features_for_window(grp, c_ts, priors_map[c_ts], time_col="target_end_ist")
        parts.append(f_grp[["ID", "diurnal_14d_med"]])

    merged = test_raw[["ID"]].merge(pd.concat(parts, ignore_index=True), on="ID", how="left")
    sub_m0 = pd.DataFrame({"ID": test_raw["ID"], "Target": merged["diurnal_14d_med"].clip(lower=0.0)})
    print(
        f"[Validation Check M0] Model 0 Heuristic Baseline ready | Rows={len(sub_m0):,} | "
        f"Mean={sub_m0['Target'].mean():.2f} | Median={sub_m0['Target'].median():.2f}"
    )
    return sub_m0
