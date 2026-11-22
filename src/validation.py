"""
Hard submission contract assertions, physics audits, immutable run archiving, and experiment logging.

Design logic:
- Prevents silent overwriting of previous submissions by creating a unique timestamped
  folder inside runs/ for every run (storing CSV, config.json, metrics.json, git_state.txt).
- Enforces every Zindi format requirement before writing a submission CSV.
- Automatically appends one clean, reproducible row to docs/02_Experiment_Log.csv.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error

from src.config import DOCS_DIR, PUB_WINDOWS, RUNS_DIR


def assert_submission_contract(
    sub_df: pd.DataFrame,
    test_raw: pd.DataFrame,
    sample_sub_raw: pd.DataFrame,
) -> None:
    """Hard contract checks guaranteeing exact competition submission format and ID order."""
    assert len(sub_df) == 2592, f"Expected 2,592 rows, got {len(sub_df)}"
    assert list(sub_df.columns) == ["ID", "Target"], f"Invalid columns: {list(sub_df.columns)}"
    assert (sub_df["ID"].values == test_raw["ID"].values).all(), "ID order mismatch vs Test.csv!"
    assert (sub_df["ID"].values == sample_sub_raw["ID"].values).all(), "ID order mismatch vs SampleSubmission.csv!"
    assert int(sub_df["Target"].isna().sum()) == 0, "NaNs detected in submission Target!"
    assert np.isfinite(sub_df["Target"].to_numpy()).all(), "Infinite values detected in Target!"
    assert (sub_df["Target"] >= 0.0).all(), "Negative predictions detected in Target!"
    print("[Validation Check 5A] Submission Contract PASSED (2,592 rows | IDs aligned | 0 NaNs | Non-negative).")


def audit_private_block_physics(sub_detail_df: pd.DataFrame, label: str = "Final") -> None:
    """
    Audits Private Leaderboard (Nov 11-12, n=192) predictions by station geography and
    diurnal physics (Nocturnal Inversion 22h-09h must exceed Daytime Sea Breeze 10h-21h).
    """
    priv = sub_detail_df[sub_detail_df["scoring_block"] == "private"].copy()
    priv["hour_int"] = priv["target_end_ist"].dt.hour
    priv["period_of_day"] = np.where(
        priv["hour_int"].isin([22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        "Nocturnal_Inversion (22h-09h)",
        "Daytime_SeaBreeze (10h-21h)",
    )

    print(f"\n[Validation Check 5B] Private LB (Nov 11-12, n={len(priv)}) Station Summary — {label}:")
    print(
        priv.groupby("location_id")
        .agg(
            m0_14d=("diurnal_14d_med", "mean"),
            hyb_base=("baseline_prior", "mean"),
            pred_mean=("pred_target", "mean"),
            pred_median=("pred_target", "median"),
            pred_max=("pred_target", "max"),
        )
        .to_string()
    )

    diurnal = priv.groupby("period_of_day").agg(
        rows=("ID", "count"),
        m0_14d_mean=("diurnal_14d_med", "mean"),
        pred_mean=("pred_target", "mean"),
    )
    print(f"\n[Validation Check 5C] Private LB Diurnal Physics Check — {label}:")
    print(diurnal.to_string())


def audit_unlocked_public_windows(
    sub_df: pd.DataFrame,
    test_raw: pd.DataFrame,
    train_raw: pd.DataFrame,
    label: str = "submission",
    pub_windows: dict[str, tuple[str, str]] = PUB_WINDOWS,
) -> dict[str, dict[str, float]]:
    """Scores a submission against all unlocked windows using both raw and cleaned ground truth."""
    test_gt = test_raw[["ID", "location_id", "target_end_ist", "scoring_block"]].merge(
        train_raw[["location_id", "timestamp_ist", "pm25_raw", "pm25"]],
        left_on=["location_id", "target_end_ist"],
        right_on=["location_id", "timestamp_ist"],
        how="left",
    )
    merged = test_gt.merge(sub_df[["ID", "Target"]], on="ID", how="left")

    summary: dict[str, dict[str, float]] = {}
    row_str = f"  {label:<32}"
    for w_name, (t_start, t_end) in pub_windows.items():
        w_mask = (merged["target_end_ist"] >= pd.Timestamp(t_start)) & (merged["target_end_ist"] <= pd.Timestamp(t_end))
        w_sub = merged[w_mask]
        v_raw = w_sub.dropna(subset=["pm25_raw", "Target"])
        v_cln = w_sub.dropna(subset=["pm25", "Target"])
        r_raw = float(root_mean_squared_error(v_raw["pm25_raw"], v_raw["Target"])) if len(v_raw) > 0 else float("nan")
        r_cln = float(root_mean_squared_error(v_cln["pm25"], v_cln["Target"])) if len(v_cln) > 0 else float("nan")
        summary[w_name] = {"raw_rmse": round(r_raw, 4), "clean_rmse": round(r_cln, 4), "n": int(len(v_raw))}
        row_str += f" | {w_name}: Raw={r_raw:6.2f} Cln={r_cln:6.2f}"

    print("\n[Validation Check 5D] Unlocked Window Audit (Raw vs Cleaned Ground Truth):")
    print(row_str)
    return summary


def get_git_state() -> dict[str, str]:
    """Captures active Git branch and short commit hash for reproducibility audit trails."""
    def _run(cmd: list[str]) -> str:
        try:
            return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:
            return "not_a_git_repo"

    return {
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "commit": _run(["git", "rev-parse", "--short", "HEAD"]),
        "dirty": "yes" if _run(["git", "status", "--porcelain"]) not in ("", "not_a_git_repo") else "no",
    }


def save_immutable_run_artifacts(
    run_label: str,
    sub_df: pd.DataFrame,
    metrics_dict: dict[str, Any],
    config_dict: dict[str, Any],
    oof_df: pd.DataFrame | None = None,
    runs_dir: Path = RUNS_DIR,
) -> Path:
    """
    Zero-Overwrite Exporter: creates a unique auto-incrementing, timestamped directory
    inside runs/ storing submission CSV, metrics.json, config.json, and git_state.txt.
    """
    runs_dir.mkdir(parents=True, exist_ok=True)
    existing_runs = [p for p in runs_dir.iterdir() if p.is_dir() and p.name.startswith("r")]
    next_id = len(existing_runs) + 1

    ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    cv_val = metrics_dict.get("pooled_cv_rmse", 0.0)
    safe_label = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in run_label)
    run_folder = runs_dir / f"r{next_id:03d}_{ts_tag}_{safe_label}_cv{cv_val:.2f}"
    run_folder.mkdir(parents=True, exist_ok=False)

    sub_path = run_folder / f"sub_r{next_id:03d}_{safe_label}_cv{cv_val:.2f}.csv"
    sub_df.to_csv(sub_path, index=False)

    (run_folder / "metrics.json").write_text(json.dumps(metrics_dict, indent=2, default=str), encoding="utf-8")
    (run_folder / "config.json").write_text(json.dumps(config_dict, indent=2, default=str), encoding="utf-8")
    (run_folder / "git_state.txt").write_text(json.dumps(get_git_state(), indent=2), encoding="utf-8")

    if oof_df is not None:
        oof_df.to_parquet(run_folder / "oof_predictions.parquet", index=False)

    print(f"[Validation Check 5E] Saved immutable run bundle (zero-overwrite): {run_folder}")
    print(f"  -> Submission CSV: {sub_path.name}")
    return sub_path


def append_experiment_log(
    row_dict: dict[str, Any],
    log_path: Path = DOCS_DIR / "02_Experiment_Log.csv",
) -> None:
    """Automatically appends one standardized audit row to docs/02_Experiment_Log.csv."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    row_df = pd.DataFrame([row_dict])
    write_header = not log_path.exists()
    row_df.to_csv(log_path, mode="a", header=write_header, index=False)
    print(f"[Validation Check 5F] Appended experiment record to {log_path}")
