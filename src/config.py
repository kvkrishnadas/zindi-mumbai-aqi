"""
Project configuration, deterministic seeds, local path resolution, and preflight checks.

Design logic:
- Uses pathlib.Path relative to the repository root so identical code runs on any local
  OS (Windows/macOS/Linux) with zero hardcoded user paths.
- Keeps all competition constants (station maps, validation windows, feature lists, and
  portfolio presets) in one central file so downstream modules never drift out of sync.
"""

from __future__ import annotations

import os
import platform
import random
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy
import sklearn

# Optional open-source tree libraries (Rule 1 compliant; wrapped safely for portability)
try:
    import xgboost as xgb

    HAS_XGB = True
    XGB_VERSION = xgb.__version__
except ImportError:
    HAS_XGB = False
    XGB_VERSION = "not installed"

try:
    import catboost

    HAS_CAT = True
    CAT_VERSION = catboost.__version__
except ImportError:
    HAS_CAT = False
    CAT_VERSION = "not installed"

# ------------------------------------------------------------------------------
# Reproducibility & Local CPU Multithreading
# ------------------------------------------------------------------------------
SEED: int = 42
MAX_CORES: int = os.cpu_count() or 4

# ------------------------------------------------------------------------------
# Project Directory Layout (Relative to Repository Root)
# ------------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
DOCS_DIR: Path = PROJECT_ROOT / "docs"
RUNS_DIR: Path = PROJECT_ROOT / "runs"

# ------------------------------------------------------------------------------
# Domain Constants & Forensic Audit Guardrails
# ------------------------------------------------------------------------------
TIMEZONE_IST: str = "Asia/Kolkata"

# Verified 2024 historical site IDs (Trap 2 defense against twin-sensor calibration offset)
VALID_2024_SITES: set[str] = {"site_5113", "site_5102", "site_5960", "site_5961"}

# Fixed integer encoding for location_id used by HGB / XGBoost / CatBoost
LOC_MAP: dict[int, int] = {
    11606: 0,    # Borivali East, Mumbai - IITM
    11611: 1,    # Malad West, Mumbai - IITM
    3409510: 2,  # Byculla, Mumbai - BMC
    3409511: 3,  # Shivaji Nagar, Mumbai - BMC
}

# Exact 24 causal, gap-safe model features
FEATURE_COLS: list[str] = [
    "location_id",
    "hour_int",
    "day_of_week",
    "is_weekend",
    "cutoff_hour",
    "hour_sin",
    "hour_cos",
    "inversion_weight",
    "horizon_hours",
    "station_72h_std",
    "slow_st_72h_anom",
    "slow_city_72h_anom",
    "slow_d5_vs_d14",
    "slow_d7_vs_d14",
    "fast_st_6h_anom",
    "fast_st_24h_anom",
    "fast_city_24h_anom",
    "fast_d3_vs_base",
    "vel_6h_vs_24h",
    "vel_24h_vs_72h",
    "city_vel_24h_vs_72h",
    "inv_x_st_72h",
    "inv_x_st_24h",
    "st_vs_city_72h",
]

# 6 Rule-2 compliant out-of-sample causal validation windows covering stationary,
# post-hole (5-day Oct 24-28 gap), regime-drop, pre-Diwali, and post-Diwali regimes.
VAL_SPECS: list[tuple[str, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = [
    (
        "W1_Oct22_23 (Stationary)",
        pd.Timestamp("2026-10-21 23:00:00+05:30"),
        pd.Timestamp("2026-10-22 01:00:00+05:30"),
        pd.Timestamp("2026-10-23 23:00:00+05:30"),
    ),
    (
        "W2_Oct27_28 (Post-5d-Hole)",
        pd.Timestamp("2026-10-23 21:00:00+05:30"),
        pd.Timestamp("2026-10-27 10:00:00+05:30"),
        pd.Timestamp("2026-10-28 23:00:00+05:30"),
    ),
    (
        "W3_Nov01_02 (Regime Drop)",
        pd.Timestamp("2026-10-31 23:00:00+05:30"),
        pd.Timestamp("2026-11-01 01:00:00+05:30"),
        pd.Timestamp("2026-11-02 23:00:00+05:30"),
    ),
    (
        "W4_Nov05_06 (From Nov04 21h)",
        pd.Timestamp("2026-11-04 21:00:00+05:30"),
        pd.Timestamp("2026-11-05 01:00:00+05:30"),
        pd.Timestamp("2026-11-06 23:00:00+05:30"),
    ),
    (
        "W5_Nov06_07 (From Nov05 09h)",
        pd.Timestamp("2026-11-05 09:00:00+05:30"),
        pd.Timestamp("2026-11-06 01:00:00+05:30"),
        pd.Timestamp("2026-11-07 23:00:00+05:30"),
    ),
    (
        "W6_Nov09_10 (Post-Diwali 09h)",
        pd.Timestamp("2026-11-08 09:00:00+05:30"),
        pd.Timestamp("2026-11-09 01:00:00+05:30"),
        pd.Timestamp("2026-11-10 09:00:00+05:30"),
    ),
]

# Unlocked public windows for retrospective Raw vs Clean diagnostic auditing
PUB_WINDOWS: dict[str, tuple[str, str]] = {
    "W1_Oct22_23": ("2026-10-22 01:00:00+05:30", "2026-10-24 00:00:00+05:30"),
    "W2_Oct27_28": ("2026-10-27 01:00:00+05:30", "2026-10-29 00:00:00+05:30"),
    "W3_Nov01_02": ("2026-11-01 01:00:00+05:30", "2026-11-03 00:00:00+05:30"),
    "W4_Nov05_06": ("2026-11-05 01:00:00+05:30", "2026-11-07 00:00:00+05:30"),
    "W5_Nov08_10_LB": ("2026-11-08 00:00:00+05:30", "2026-11-10 09:00:00+05:30"),
}

# Explicit causal cutoffs mirroring the rolling Train.csv release schedule
BASE_CAUSAL_CUTOFFS: list[pd.Timestamp] = [
    pd.Timestamp("2026-10-21 23:00:00+05:30"),
    pd.Timestamp("2026-10-23 21:00:00+05:30"),
    pd.Timestamp("2026-10-31 23:00:00+05:30"),
    pd.Timestamp("2026-11-03 12:00:00+05:30"),
    pd.Timestamp("2026-11-04 23:00:00+05:30"),
    pd.Timestamp("2026-11-06 23:00:00+05:30"),
]

# Default modeling hyperparameters verified via 6-Window CV Tournament
PIPELINE_CONFIG: dict[str, Any] = {
    "seed": SEED,
    "clip_cap": 190.0,
    "mean_blend": 0.45,
    "prior_weights": (0.45, 0.35, 0.20),
    "tau_fast": 30.0,
    "tau_slow": 120.0,
    "align_method_2024": "hour_med_iqr",
    "hist_2024_months": (10, 11, 12),
    "hist_2024_sample_weight": 0.25,
    "candidate_log_weights": (0.35, 0.50, 0.65, 0.75, 0.85, 0.90, 1.00),
    "benchmark_cv_rmse": 24.6137,
}

# Structurally hedged final portfolio presets (Private LB Rank 28 architecture)
PORTFOLIO_PRESETS: dict[str, dict[str, Any]] = {
    "slot1_balanced": {
        "clip_cap": 190.0,
        "mean_blend": 0.45,
        "prior_weights": (0.45, 0.35, 0.20),
        "tau_fast": 30.0,
        "tau_slow": 120.0,
        "res_scale": 1.00,
    },
    "slot2_inversion_hedge": {
        "clip_cap": 230.0,
        "mean_blend": 0.55,
        "prior_weights": (0.25, 0.35, 0.40),
        "tau_fast": 42.0,
        "tau_slow": 160.0,
        "res_scale": 1.08,
    },
}


def set_global_seed(seed: int = SEED) -> None:
    """Locks random seeds across Python and NumPy for strict reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def resolve_raw_data_dir() -> Path:
    """
    Locates the local directory containing Test.csv.
    Checks data/raw/ first (standard repo layout), then falls back to data/ or project root.
    """
    for folder in (RAW_DATA_DIR, DATA_DIR, PROJECT_ROOT):
        if (folder / "Test.csv").is_file():
            return folder
    matches = list(PROJECT_ROOT.glob("**/Test.csv"))
    if matches:
        return matches[0].parent
    raise FileNotFoundError(
        f"Could not find Test.csv under {RAW_DATA_DIR} or {PROJECT_ROOT}. "
        "Place competition CSVs into data/raw/ before running."
    )


def run_preflight_check() -> dict[str, Any]:
    """
    Block 0 Preflight Check: locks seeds, ensures output folders exist, prints
    exact library versions, and verifies local raw data accessibility.
    """
    set_global_seed(SEED)
    for directory in (RAW_DATA_DIR, PROCESSED_DATA_DIR, DOCS_DIR, RUNS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    raw_dir = resolve_raw_data_dir()
    env_info = {
        "python": platform.python_version(),
        "os": f"{platform.system()} {platform.release()}",
        "cpu_cores": MAX_CORES,
        "seed": SEED,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "lightgbm": lgb.__version__,
        "xgboost": XGB_VERSION,
        "catboost": CAT_VERSION,
        "raw_data_dir": str(raw_dir),
    }

    print("[Validation Check 0] Local Environment & Dependency Preflight:")
    print(f"  -> Python {env_info['python']} on {env_info['os']} | CPU Cores={MAX_CORES} | SEED={SEED}")
    print(
        f"  -> pandas={env_info['pandas']} | numpy={env_info['numpy']} | "
        f"sklearn={env_info['scikit-learn']} | scipy={env_info['scipy']}"
    )
    print(
        f"  -> lightgbm={env_info['lightgbm']} | xgboost={env_info['xgboost']} "
        f"(HAS_XGB={HAS_XGB}) | catboost={env_info['catboost']} (HAS_CAT={HAS_CAT})"
    )
    print(f"  -> Resolved Raw Data Directory: {raw_dir}")
    return env_info
