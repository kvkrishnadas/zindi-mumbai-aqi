"""
Pure-CPU multi-branch GBDT ensemble, horizon-regime gating, and 6-window causal CV tournament.

Design logic:
- Uses only open-source, deterministic CPU tree learners (LightGBM, Scikit-Learn HGB,
  optional XGBoost and CatBoost) with fixed global SEED = 42.
- Pairs Additive Residual branches (strong on stationary & post-hole long horizons > 64h)
  with Multiplicative Log-Ratio branches + Duan's Smearing (strong on right-skewed winter
  inversions and post-Diwali transitions <= 64h).
- Evaluates 6 strictly causal out-of-sample windows (W1-W6) and reports both pooled RMSE
  and non-firework W1-W5 RMSE to prevent single-window overfitting.
"""

from __future__ import annotations

import time
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error

from src.config import FEATURE_COLS, HAS_CAT, HAS_XGB, LOC_MAP, MAX_CORES, SEED, VAL_SPECS
from src.features import build_features_for_window, compute_cutoff_priors

if HAS_XGB:
    from xgboost import XGBRegressor
if HAS_CAT:
    from catboost import CatBoostRegressor


def prep_num_features(df_in: pd.DataFrame) -> pd.DataFrame:
    """Encodes location_id as deterministic integer indices for HGB / XGBoost / CatBoost."""
    out = df_in[FEATURE_COLS].copy()
    out["location_id"] = out["location_id"].astype(int).map(LOC_MAP).fillna(0).astype(int)
    return out


def fit_multibranch_ensemble(
    tr_df: pd.DataFrame,
    sample_weight: np.ndarray | None = None,
    include_xgb_cat: bool = True,
    n_jobs: int = MAX_CORES,
) -> dict[str, Any]:
    """
    Fits symmetric CPU-multithreaded Additive and Multiplicative Log-Ratio branches:
    - Additive (residual over baseline_prior): LGBM_Huber, LGBM_L2, HGB, (+ XGB_Add, Cat_Add)
    - Multiplicative (log-ratio + Duan's smearing): LGBM_Log, (+ XGB_Log, Cat_Log)
    """
    y_res = tr_df["target_residual"].clip(-50.0, 95.0)
    y_log = tr_df["target_log_ratio"].clip(-0.85, 1.10)
    X_num = prep_num_features(tr_df)
    mask_25 = (tr_df["is_2024_hist"].to_numpy() == 0) if "is_2024_hist" in tr_df.columns else np.ones(len(tr_df), dtype=bool)

    m_huber = lgb.LGBMRegressor(
        objective="huber", alpha=14.0, metric="rmse", learning_rate=0.025,
        num_leaves=15, max_depth=4, min_child_samples=120, subsample=0.8,
        subsample_freq=1, colsample_bytree=0.8, reg_alpha=8.0, reg_lambda=30.0,
        n_estimators=190, random_state=SEED, n_jobs=n_jobs, verbose=-1,
    )
    m_huber.fit(tr_df[FEATURE_COLS], y_res, sample_weight=sample_weight, categorical_feature=["location_id"])

    m_l2 = lgb.LGBMRegressor(
        objective="regression", metric="rmse", learning_rate=0.02,
        num_leaves=12, max_depth=4, min_child_samples=150, subsample=0.75,
        subsample_freq=1, colsample_bytree=0.75, reg_alpha=15.0, reg_lambda=50.0,
        n_estimators=170, random_state=SEED + 100, n_jobs=n_jobs, verbose=-1,
    )
    m_l2.fit(tr_df[FEATURE_COLS], y_res, sample_weight=sample_weight, categorical_feature=["location_id"])

    m_hgb = HistGradientBoostingRegressor(
        loss="squared_error", max_iter=160, learning_rate=0.03, max_leaf_nodes=15,
        max_depth=4, min_samples_leaf=120, l2_regularization=20.0,
        categorical_features=[0], random_state=SEED + 300,
    )
    m_hgb.fit(X_num, y_res, sample_weight=sample_weight)

    m_log_lgb = lgb.LGBMRegressor(
        objective="regression", metric="rmse", learning_rate=0.025,
        num_leaves=15, max_depth=4, min_child_samples=130, subsample=0.8,
        subsample_freq=1, colsample_bytree=0.8, reg_alpha=5.0, reg_lambda=25.0,
        n_estimators=180, random_state=SEED + 200, n_jobs=n_jobs, verbose=-1,
    )
    m_log_lgb.fit(tr_df[FEATURE_COLS], y_log, sample_weight=sample_weight, categorical_feature=["location_id"])
    smear_lgb = float(np.clip(np.mean(np.exp(y_log.to_numpy()[mask_25] - m_log_lgb.predict(tr_df[FEATURE_COLS])[mask_25])), 1.00, 1.06))

    models: dict[str, Any] = {
        "m_huber": m_huber, "m_l2": m_l2, "m_hgb": m_hgb,
        "m_log_lgb": m_log_lgb, "smear_lgb": smear_lgb,
        "include_xgb_cat": include_xgb_cat,
    }

    if include_xgb_cat and HAS_XGB:
        m_xgb_add = XGBRegressor(
            n_estimators=175, max_depth=4, learning_rate=0.025, subsample=0.8,
            colsample_bytree=0.8, reg_alpha=10.0, reg_lambda=35.0, min_child_weight=120,
            tree_method="hist", random_state=SEED + 400, n_jobs=n_jobs, verbosity=0,
        )
        m_xgb_add.fit(X_num, y_res, sample_weight=sample_weight)
        models["m_xgb_add"] = m_xgb_add

        m_xgb_log = XGBRegressor(
            n_estimators=180, max_depth=4, learning_rate=0.025, subsample=0.8,
            colsample_bytree=0.8, reg_alpha=5.0, reg_lambda=25.0, min_child_weight=120,
            tree_method="hist", random_state=SEED + 410, n_jobs=n_jobs, verbosity=0,
        )
        m_xgb_log.fit(X_num, y_log, sample_weight=sample_weight)
        smear_xgb = float(np.clip(np.mean(np.exp(y_log.to_numpy()[mask_25] - m_xgb_log.predict(X_num)[mask_25])), 1.00, 1.06))
        models["m_xgb_log"] = m_xgb_log
        models["smear_xgb"] = smear_xgb

    if include_xgb_cat and HAS_CAT:
        m_cat_add = CatBoostRegressor(
            iterations=210, depth=4, learning_rate=0.03, l2_leaf_reg=30.0,
            task_type="CPU", thread_count=n_jobs, random_seed=SEED + 500, verbose=0,
        )
        m_cat_add.fit(X_num, y_res, sample_weight=sample_weight)
        models["m_cat_add"] = m_cat_add

        m_cat_log = CatBoostRegressor(
            iterations=220, depth=4, learning_rate=0.03, l2_leaf_reg=25.0,
            task_type="CPU", thread_count=n_jobs, random_seed=SEED + 510, verbose=0,
        )
        m_cat_log.fit(X_num, y_log, sample_weight=sample_weight)
        smear_cat = float(np.clip(np.mean(np.exp(y_log.to_numpy()[mask_25] - m_cat_log.predict(X_num)[mask_25])), 1.00, 1.06))
        models["m_cat_log"] = m_cat_log
        models["smear_cat"] = smear_cat

    return models


def predict_multibranch_ensemble(
    models: dict[str, Any],
    val_df: pd.DataFrame,
    res_scale: float = 1.0,
) -> dict[str, np.ndarray]:
    """Generates both Additive and Multiplicative Log-Ratio branch predictions."""
    base_arr = val_df["baseline_prior"].to_numpy()
    X_num = prep_num_features(val_df)

    add_preds = [
        base_arr + res_scale * models["m_huber"].predict(val_df[FEATURE_COLS]),
        base_arr + res_scale * models["m_l2"].predict(val_df[FEATURE_COLS]),
        base_arr + res_scale * models["m_hgb"].predict(X_num),
    ]
    add_weights = [0.45, 0.30, 0.25]
    if models.get("include_xgb_cat", False) and "m_xgb_add" in models:
        add_preds.append(base_arr + res_scale * models["m_xgb_add"].predict(X_num))
        add_weights.append(0.20)
    if models.get("include_xgb_cat", False) and "m_cat_add" in models:
        add_preds.append(base_arr + res_scale * models["m_cat_add"].predict(X_num))
        add_weights.append(0.20)

    w_add = np.array(add_weights) / np.sum(add_weights)
    pred_add = np.clip(sum(w * p for w, p in zip(w_add, add_preds)), 0.0, None)

    z_lgb = res_scale * models["m_log_lgb"].predict(val_df[FEATURE_COLS])
    log_preds = [(base_arr + 10.0) * np.exp(z_lgb) * models["smear_lgb"] - 10.0]
    log_weights = [0.60]
    if models.get("include_xgb_cat", False) and "m_xgb_log" in models:
        z_xgb = res_scale * models["m_xgb_log"].predict(X_num)
        log_preds.append((base_arr + 10.0) * np.exp(z_xgb) * models["smear_xgb"] - 10.0)
        log_weights.append(0.20)
    if models.get("include_xgb_cat", False) and "m_cat_log" in models:
        z_cat = res_scale * models["m_cat_log"].predict(X_num)
        log_preds.append((base_arr + 10.0) * np.exp(z_cat) * models["smear_cat"] - 10.0)
        log_weights.append(0.20)

    w_log = np.array(log_weights) / np.sum(log_weights)
    pred_log_mult = np.clip(sum(w * p for w, p in zip(w_log, log_preds)), 0.0, None)
    return {"pred_add": pred_add, "pred_log_mult": pred_log_mult}


def compute_gated_log_weight(
    df_in: pd.DataFrame,
    params: tuple[float, float, float] = (1.0, 0.0, 0.5),
) -> np.ndarray:
    """
    Horizon & Regime Gate controlling the blend between Additive and Multiplicative branches:
    - Long horizons (> 64h across multi-day holes, e.g. W2): uses w_long (favors Additive).
    - Cooling stationary regime (3d prior < baseline and calm city anomaly): uses w_cool.
    - Active winter / post-Diwali transition (<= 64h): uses w_active (favors Multiplicative).
    """
    w_active, w_long, w_cool = params
    raw_h = df_in["raw_horizon_hours"].to_numpy() if "raw_horizon_hours" in df_in.columns else df_in["horizon_hours"].to_numpy()
    d3_vs_base = df_in["fast_d3_vs_base"].to_numpy()
    city_24h = df_in["fast_city_24h_anom"].to_numpy()

    is_long_gap = raw_h > 64.0
    is_cooling = (~is_long_gap) & (d3_vs_base < -0.5) & (city_24h <= 0.5)
    return np.clip(np.where(is_long_gap, w_long, np.where(is_cooling, w_cool, w_active)), 0.0, 1.0)


def evaluate_single_cv_mode(
    mode_name: str,
    sup_df: pd.DataFrame,
    train_clean: pd.DataFrame,
    include_xgb_cat: bool = True,
    hist_2024_weight: float = 0.25,
    candidate_log_weights: tuple[float, ...] = (0.35, 0.50, 0.65, 0.75, 0.85, 0.90, 1.00),
    val_specs: list[tuple[str, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = VAL_SPECS,
) -> dict[str, Any]:
    """Runs strictly causal 6-window out-of-sample CV for one candidate configuration."""
    t0 = time.perf_counter()
    oof_records = []

    for w_label, c_ts, t_start, t_end in val_specs:
        tr_sub = sup_df[sup_df["timestamp_ist"] <= c_ts].copy()
        val_raw = train_clean[(train_clean["timestamp_ist"] >= t_start) & (train_clean["timestamp_ist"] <= t_end)].copy()
        if len(val_raw) == 0:
            continue

        val_priors = compute_cutoff_priors(train_clean, c_ts, clip_cap=190.0, mean_blend=0.45)
        val_df = build_features_for_window(val_raw, c_ts, val_priors, time_col="timestamp_ist")
        val_df["location_id"] = val_df["location_id"].astype("category")

        sw = None
        if "is_2024_hist" in tr_sub.columns and (tr_sub["is_2024_hist"] == 1).any():
            sw = np.where(tr_sub["is_2024_hist"].to_numpy() == 1, hist_2024_weight, 1.0)

        models = fit_multibranch_ensemble(tr_sub, sample_weight=sw, include_xgb_cat=include_xgb_cat)
        preds = predict_multibranch_ensemble(models, val_df)
        val_df["pred_add"] = preds["pred_add"]
        val_df["pred_log_mult"] = preds["pred_log_mult"]
        val_df["window_label"] = w_label
        oof_records.append(val_df)

    oof_all = pd.concat(oof_records, ignore_index=True)

    # Compare static log_weight candidates vs Nelder-Mead optimized horizon-regime gating
    best_scheme, best_rmse, best_lw = "static_1.00", 999.0, 1.00
    for lw in candidate_log_weights:
        cand_pred = np.clip((1.0 - lw) * oof_all["pred_add"] + lw * oof_all["pred_log_mult"], 0.0, None)
        rmse = root_mean_squared_error(oof_all["pm25"], cand_pred)
        if rmse < best_rmse:
            best_rmse, best_scheme, best_lw = rmse, f"static_{lw:.2f}", lw

    def _gate_loss(p_vec: np.ndarray) -> float:
        lw_v = compute_gated_log_weight(oof_all, params=tuple(np.clip(p_vec, 0.0, 1.0)))
        pr = (1.0 - lw_v) * oof_all["pred_add"].to_numpy() + lw_v * oof_all["pred_log_mult"].to_numpy()
        return root_mean_squared_error(oof_all["pm25"].to_numpy(), pr)

    opt = minimize(_gate_loss, x0=[0.98, 0.05, 0.15], bounds=[(0.5, 1.0), (0.0, 0.5), (0.0, 0.5)], method="Nelder-Mead")
    gated_params = tuple(float(x) for x in np.clip(opt.x, 0.0, 1.0))
    lw_gated = compute_gated_log_weight(oof_all, params=gated_params)
    pred_gated = np.clip((1.0 - lw_gated) * oof_all["pred_add"] + lw_gated * oof_all["pred_log_mult"], 0.0, None)
    rmse_gated = root_mean_squared_error(oof_all["pm25"], pred_gated)

    if rmse_gated < best_rmse:
        best_rmse, best_scheme = rmse_gated, "horizon_regime_gated"

    oof_all["pred_ens_best"] = (
        pred_gated
        if best_scheme == "horizon_regime_gated"
        else np.clip((1.0 - best_lw) * oof_all["pred_add"] + best_lw * oof_all["pred_log_mult"], 0.0, None)
    )

    rmse_add = root_mean_squared_error(oof_all["pm25"], oof_all["pred_add"])
    rmse_log = root_mean_squared_error(oof_all["pm25"], oof_all["pred_log_mult"])
    oof_w1_w5 = oof_all[oof_all["window_label"] != "W6_Nov09_10 (Post-Diwali 09h)"]
    rmse_w1_w5 = root_mean_squared_error(oof_w1_w5["pm25"], oof_w1_w5["pred_ens_best"])
    elapsed = time.perf_counter() - t0

    print(
        f"  [{mode_name:<34}] ({elapsed:5.1f}s) | Add: {rmse_add:.4f} | LogM: {rmse_log:.4f} | "
        f"Gated: {rmse_gated:.4f} | Best ({best_scheme}): {best_rmse:.4f} | W1-W5: {rmse_w1_w5:.4f}"
    )
    return {
        "mode_name": mode_name,
        "sup_df": sup_df,
        "include_xgb_cat": include_xgb_cat,
        "best_scheme_name": best_scheme,
        "best_lw_param": best_lw,
        "gated_params": gated_params,
        "best_rmse": float(best_rmse),
        "rmse_add": float(rmse_add),
        "rmse_log": float(rmse_log),
        "rmse_w1_w5": float(rmse_w1_w5),
        "runtime_sec": round(elapsed, 2),
        "oof_all": oof_all,
    }


def run_6window_cv_tournament(
    candidate_modes: list[tuple[str, pd.DataFrame, bool]],
    train_clean: pd.DataFrame,
    hist_2024_weight: float = 0.25,
    benchmark_rmse: float = 24.6137,
) -> dict[str, Any]:
    """Runs all candidate modes through the 6-window gate and prints the winner breakdown."""
    print("\n==================================================================================================")
    print("[Validation Check 4A] 6-Window Out-of-Sample Causal CV Tournament (Local CPU):")
    print("==================================================================================================")
    results = [
        evaluate_single_cv_mode(name, s_df, train_clean, include_xgb_cat=use_xc, hist_2024_weight=hist_2024_weight)
        for name, s_df, use_xc in candidate_modes
    ]
    winner = min(results, key=lambda r: r["best_rmse"])

    print("==================================================================================================")
    print(f"[Validation Gate] Winning Mode            : {winner['mode_name']}")
    print(f"[Validation Gate] Winning Blend Scheme    : {winner['best_scheme_name']} | Params={[round(x, 3) for x in winner['gated_params']]}")
    print(f"[Validation Gate] Pooled 6-Window RMSE    : {winner['best_rmse']:.4f} ug/m3 (vs Bench {benchmark_rmse:.4f}: {winner['best_rmse'] - benchmark_rmse:+.4f})")
    print(f"[Validation Gate] W1-W5 Non-Firework RMSE : {winner['rmse_w1_w5']:.4f} ug/m3")
    print("==================================================================================================")

    print(f"\n[Validation Check 4B] Per-Window Breakdown for '{winner['mode_name']}':")
    for w_label, grp in winner["oof_all"].groupby("window_label", sort=False):
        m0_r = root_mean_squared_error(grp["pm25"], grp["diurnal_14d_med"])
        bs_r = root_mean_squared_error(grp["pm25"], grp["baseline_prior"])
        ad_r = root_mean_squared_error(grp["pm25"], grp["pred_add"])
        lg_r = root_mean_squared_error(grp["pm25"], grp["pred_log_mult"])
        en_r = root_mean_squared_error(grp["pm25"], grp["pred_ens_best"])
        print(
            f"  -> {w_label:<29} | Act:{grp['pm25'].mean():5.1f} | M0:{m0_r:6.2f} | Base:{bs_r:6.2f} "
            f"| Add:{ad_r:6.2f} | LogM:{lg_r:6.2f} | BestHyb:{en_r:6.2f} (Mean:{grp['pred_ens_best'].mean():5.1f})"
        )

    return {"winner": winner, "all_results": results}


def generate_test_predictions(
    full_models: dict[str, Any],
    train_clean: pd.DataFrame,
    test_raw: pd.DataFrame,
    causal_cutoffs: list[pd.Timestamp],
    clip_cap: float = 190.0,
    mean_blend: float = 0.45,
    prior_weights: tuple[float, float, float] = (0.45, 0.35, 0.20),
    tau_fast: float = 30.0,
    tau_slow: float = 120.0,
    scheme_name: str = "horizon_regime_gated",
    lw_param: float = 1.0,
    gated_params: tuple[float, float, float] = (1.0, 0.0, 0.5),
    res_scale: float = 1.0,
) -> pd.DataFrame:
    """Produces strictly causal multi-branch test predictions preserving exact Test.csv ID order."""
    priors_map = {
        c_ts: compute_cutoff_priors(train_clean, c_ts, clip_cap=clip_cap, mean_blend=mean_blend)
        for c_ts in causal_cutoffs
    }
    test_work = test_raw[["ID", "location_id", "target_end_ist", "scoring_block"]].copy()
    test_work["assigned_cutoff"] = test_work["target_end_ist"].apply(
        lambda ts: [c for c in causal_cutoffs if c < ts][-1] if any(c < ts for c in causal_cutoffs) else causal_cutoffs[0]
    )

    parts = []
    for c_ts, grp in test_work.groupby("assigned_cutoff", sort=False):
        f_grp = build_features_for_window(
            grp, c_ts, priors_map[c_ts], time_col="target_end_ist",
            prior_weights=prior_weights, tau_fast=tau_fast, tau_slow=tau_slow,
        )
        f_grp["location_id"] = f_grp["location_id"].astype("category")
        br_preds = predict_multibranch_ensemble(full_models, f_grp, res_scale=res_scale)
        lw_vec = compute_gated_log_weight(f_grp, params=gated_params) if scheme_name == "horizon_regime_gated" else lw_param
        f_grp["pred_target"] = np.clip((1.0 - lw_vec) * br_preds["pred_add"] + lw_vec * br_preds["pred_log_mult"], 0.0, None)
        parts.append(
            f_grp[["ID", "location_id", "target_end_ist", "scoring_block", "assigned_cutoff", "diurnal_14d_med", "baseline_prior", "pred_target"]]
        )

    merged_all = pd.concat(parts, ignore_index=True)
    return test_raw[["ID", "location_id", "target_end_ist", "scoring_block"]].merge(
        merged_all[["ID", "assigned_cutoff", "diurnal_14d_med", "baseline_prior", "pred_target"]],
        on="ID",
        how="left",
    )
