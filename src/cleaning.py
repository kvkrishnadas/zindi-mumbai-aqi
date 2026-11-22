"""
Data loading, two-stage hardware/spatial anomaly scrubbing, and historical alignment.

Design logic:
- Follows semantic stage naming: train_raw and test_raw are NEVER mutated in place.
- Each helper performs one focused job (~20-35 lines) and accepts an optional
  `trace: dict | None = None` dictionary so intermediate tables can be inspected
  interactively in VS Code Data Wrangler without changing normal return signatures.
- Enforces Forensic Audit Rules:
  1. Scrub CPCB BAM hardware sentinel codes (flag_high_1000 == 1, > 450, < 2.0).
  2. Scrub isolated single-station spatial spikes (> 240 and > 3.5x hourly city median).
  3. Explicitly exclude Legacy_Observations_2020_2022.csv.gz.
  4. Align History_Mumbai_2024 sub-hourly (:30:00) timestamps to top-of-hour (:00:00),
     filter strictly by exact station_name (Trap 1) and verified site IDs (Trap 2),
     and apply per-station-hour Median-IQR calibration against 2025-2026 reference data.
"""

from __future__ import annotations

import gzip
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import PROCESSED_DATA_DIR, TIMEZONE_IST, VALID_2024_SITES


def find_latest_train_file(raw_dir: Path) -> Path:
    """
    Audits all Train*.csv files in raw_dir (excluding auxiliary history files) and
    selects the file with the latest validated timestamp_ist.
    """
    candidates = sorted(
        p
        for p in raw_dir.glob("*.csv")
        if p.name.lower().startswith("train") and "auxiliary" not in p.name.lower()
    )
    if not candidates:
        raise FileNotFoundError(f"No Train*.csv found in {raw_dir}")

    best_path, best_max_ts, best_rows = None, None, 0
    for cand in candidates:
        tmp = pd.read_csv(cand, usecols=["timestamp_ist"])
        ts = pd.to_datetime(tmp["timestamp_ist"], utc=True).dt.tz_convert(TIMEZONE_IST)
        cand_max, cand_rows = ts.max(), len(tmp)
        if (best_max_ts is None) or (cand_max > best_max_ts) or (cand_max == best_max_ts and cand_rows > best_rows):
            best_path, best_max_ts, best_rows = cand, cand_max, cand_rows

    print(
        f"[Validation Check 1A] Selected latest training file: '{best_path.name}' "
        f"(Rows={best_rows:,} | Max TS={best_max_ts})"
    )
    return best_path


def load_raw_core_datasets(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """
    Loads Train, Test, and SampleSubmission into immutable raw DataFrames with IST timezones.
    Preserves `pm25_raw` for retrospective Public LB verification and drops duplicate `mean`.
    """
    train_path = find_latest_train_file(raw_dir)
    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(raw_dir / "Test.csv")
    sample_sub_raw = pd.read_csv(raw_dir / "SampleSubmission.csv")

    train_raw["timestamp_ist"] = pd.to_datetime(train_raw["timestamp_ist"], utc=True).dt.tz_convert(TIMEZONE_IST)
    test_raw["target_end_ist"] = pd.to_datetime(test_raw["target_end_ist"], utc=True).dt.tz_convert(TIMEZONE_IST)
    if "target_start_ist" in test_raw.columns:
        test_raw["target_start_ist"] = pd.to_datetime(test_raw["target_start_ist"], utc=True).dt.tz_convert(TIMEZONE_IST)

    # Preserve unscrubbed target copy; drop redundant `mean` column if present
    train_raw["pm25_raw"] = train_raw["pm25"].copy()
    if "mean" in train_raw.columns:
        train_raw = train_raw.drop(columns=["mean"])

    exact_station_map = (
        train_raw[["station_name", "location_id"]]
        .drop_duplicates()
        .set_index("station_name")["location_id"]
        .to_dict()
    )

    print(
        f"[Validation Check 1B] Loaded raw datasets | train_raw={train_raw.shape} | "
        f"test_raw={test_raw.shape} | sample_sub={sample_sub_raw.shape} | stations={len(exact_station_map)}"
    )
    return train_raw, test_raw, sample_sub_raw, exact_station_map


def apply_stage1_hardware_scrub(
    df_in: pd.DataFrame,
    trace: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Stage 1 Scrubbing: Removes CPCB BAM hardware sentinel codes (flag_high_1000 == 1 or
    pm25 > 450.0) and post-reboot sensor zero dropouts (pm25 < 2.0).
    """
    df_hw = df_in.copy()
    hw_mask = (df_hw["pm25"] > 450.0) | (df_hw["pm25"] < 2.0)
    if "flag_high_1000" in df_hw.columns:
        hw_mask = hw_mask | (df_hw["flag_high_1000"] == 1)

    n_hw = int(hw_mask.sum())
    if trace is not None:
        trace["stage1_flagged_hw_rows"] = df_hw.loc[hw_mask].copy()

    df_hw.loc[hw_mask, "pm25"] = np.nan
    print(
        f"[Validation Check 1C] Stage 1 Hardware Scrub | Rows={len(df_hw):,} | "
        f"Flagged={n_hw:,} | Remaining Non-Null={int(df_hw['pm25'].notna().sum()):,}"
    )
    return df_hw


def apply_stage2_spatial_scrub(
    df_hw: pd.DataFrame,
    aux_clean: pd.DataFrame | None = None,
    safe_aux: bool = True,
    trace: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Stage 2 Scrubbing: Removes isolated single-station spatial anomalies where a reading
    exceeds 240 ug/m3 AND exceeds 3.5x the concurrent hourly Mumbai city median.

    Why `safe_aux=True`: Auxiliary Mumbai stations have a lower overall mean (~24 ug/m3).
    Taking max(4-station median, pooled median) whenever >= 2 target stations report prevents
    over-scrubbing legitimate winter inversions while protecting single-station hours.
    """
    df_sp = df_hw.copy()
    med_4st = df_sp.groupby("timestamp_ist")["pm25"].transform("median")
    cnt_4st = df_sp.groupby("timestamp_ist")["pm25"].transform("count")

    if aux_clean is not None and len(aux_clean) > 0:
        pooled = pd.concat(
            [df_sp[["timestamp_ist", "pm25"]].dropna(), aux_clean[["timestamp_ist", "pm25"]].dropna()],
            ignore_index=True,
        )
        cross_med_map = pooled.groupby("timestamp_ist")["pm25"].median()
        med_aux = df_sp["timestamp_ist"].map(cross_med_map).fillna(med_4st)
        city_med = np.where(cnt_4st >= 2, np.maximum(med_4st, med_aux), med_aux) if safe_aux else med_aux
    else:
        city_med = med_4st

    spatial_mask = (df_sp["pm25"] > 240.0) & (df_sp["pm25"] > (3.5 * city_med))
    n_sp = int(spatial_mask.sum())

    if trace is not None:
        trace["stage2_flagged_spatial_rows"] = df_sp.loc[spatial_mask].copy()

    df_sp.loc[spatial_mask, "pm25"] = np.nan
    train_clean = df_sp.dropna(subset=["pm25"]).sort_values("timestamp_ist").reset_index(drop=True)

    # Hard quality assertions after cleaning
    assert train_clean["pm25"].min() >= 2.0, "Stage 1 failed: values < 2.0 remain!"
    assert train_clean["pm25"].max() <= 450.0, "Stage 1 failed: values > 450.0 remain!"

    print(
        f"[Validation Check 1D] Stage 2 Spatial Scrub  | Flagged={n_sp:,} | "
        f"train_clean={train_clean.shape} | DeltaRows={len(train_clean) - len(df_hw):+,} | "
        f"Range=[{train_clean['pm25'].min():.2f}, {train_clean['pm25'].max():.2f}]"
    )
    return train_clean


def load_historical_archive(raw_dir: Path, keyword_list: list[str]) -> pd.DataFrame | None:
    """
    Loads an optional historical dataset (.csv, .csv.gz, or inside AdditionalHistory.zip).
    Explicitly refuses to load Legacy_Observations_2020_2022 or 2015_2020 files.
    """
    banned_tokens = ("2020_2022", "legacy", "2015_2020")

    for p in sorted(raw_dir.glob("**/*")):
        if not p.is_file():
            continue
        lname = p.name.lower()
        if any(b in lname for b in banned_tokens):
            continue
        if p.suffix in (".csv", ".gz") and any(k.lower() in lname for k in keyword_list):
            print(f"  -> Loaded direct historical file: {p.name}")
            return pd.read_csv(p)

    for zpath in sorted(raw_dir.glob("**/*.zip")):
        with zipfile.ZipFile(zpath, "r") as zf:
            for member in zf.namelist():
                lm = member.lower()
                if any(b in lm for b in banned_tokens):
                    continue
                if any(k.lower() in lm for k in keyword_list):
                    print(f"  -> Extracted '{member}' from '{zpath.name}'")
                    raw_bytes = zf.read(member)
                    if member.endswith(".gz"):
                        with gzip.GzipFile(fileobj=BytesIO(raw_bytes)) as gz:
                            return pd.read_csv(gz)
                    return pd.read_csv(BytesIO(raw_bytes))
    return None


def standardize_historical_df(
    df_raw: pd.DataFrame,
    source_label: str,
    ceil_to_hour: bool = False,
) -> pd.DataFrame:
    """
    Standardizes timestamp_ist and pm25 schema across historical datasets and applies
    Stage 1 + Stage 2 scrubbing. When ceil_to_hour=True, aligns :30:00 sub-hourly rows
    to the :00:00 end-of-hour convention used in Train.csv and Test.csv.
    """
    df = df_raw.copy()
    ts_col = next((c for c in ("timestamp_ist", "timestamp", "datetime", "date_ist", "time") if c in df.columns), None)
    pm_col = next((c for c in ("pm25", "mean", "value", "pm2_5") if c in df.columns), None)
    if ts_col is None or pm_col is None:
        raise KeyError(f"Missing timestamp or pm25 column in {source_label}: {list(df.columns)}")

    df["timestamp_ist"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce").dt.tz_convert(TIMEZONE_IST)
    if ceil_to_hour:
        n_subhr = int((df["timestamp_ist"].dt.minute != 0).sum())
        df["timestamp_ist"] = df["timestamp_ist"].dt.ceil("h")
        print(f"  -> [{source_label}] Ceiled {n_subhr:,} sub-hourly (:30:00) rows to top-of-hour (:00:00).")

    df["pm25"] = pd.to_numeric(df[pm_col], errors="coerce")
    hw_m = (df["pm25"] > 450.0) | (df["pm25"] < 2.0)
    if "flag_high_1000" in df.columns:
        hw_m = hw_m | (df["flag_high_1000"] == 1)
    df.loc[hw_m, "pm25"] = np.nan

    h_med = df.groupby("timestamp_ist")["pm25"].transform("median")
    sp_m = (df["pm25"] > 240.0) & (df["pm25"] > (3.5 * h_med))
    df.loc[sp_m, "pm25"] = np.nan

    df_clean = df.dropna(subset=["timestamp_ist", "pm25"]).sort_values("timestamp_ist").reset_index(drop=True)
    print(
        f"[Validation Check 2A] '{source_label}' cleaned: {len(df_clean):,} rows "
        f"(hw={int(hw_m.sum())}, sp={int(sp_m.sum())} | {df_clean['timestamp_ist'].min()} -> {df_clean['timestamp_ist'].max()})"
    )
    return df_clean


def align_history_2024(
    raw_hist_2024: pd.DataFrame,
    train_clean: pd.DataFrame,
    exact_station_map: dict[str, int],
    valid_sites: set[str] = VALID_2024_SITES,
    months: tuple[int, ...] | None = (10, 11, 12),
    align_method: str = "hour_med_iqr",
    trace: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Filters and calibrates History_Mumbai_2024 to match the 2025-2026 sensor distribution:
    - Trap 1: Exact station_name filtering (never substring matching).
    - Trap 2: Restrict twin sensors to verified site IDs + Median-IQR affine alignment.
    - Temporal isolation: Drops any rows within 20 days of Train.csv start.
    """
    h24 = standardize_historical_df(raw_hist_2024, "History_Mumbai_2024", ceil_to_hour=True)

    # Trap 1: Exact station_name match
    if "station_name" in h24.columns:
        before_st = len(h24)
        h24 = h24[h24["station_name"].isin(exact_station_map.keys())].copy()
        h24["location_id"] = h24["station_name"].map(exact_station_map).astype(int)
        print(f"[Validation Check 2B - Trap 1] Exact station_name filter: {before_st:,} -> {len(h24):,} rows.")

    # Trap 2: Filter twin IDs to verified site IDs
    for col in ("series_id", "site_id", "sensor_id", "id", "station_code"):
        if col not in h24.columns:
            continue
        col_str = h24[col].astype(str).str.strip()
        mask_sites = col_str.isin(valid_sites) | col_str.apply(lambda x: any(s in x for s in valid_sites))
        if mask_sites.sum() > 0:
            before_site = len(h24)
            h24 = h24[mask_sites].copy()
            print(f"[Validation Check 2C - Trap 2] Filtered twin IDs via '{col}': {before_site:,} -> {len(h24):,} rows.")
            break

    h24 = h24.groupby(["location_id", "timestamp_ist"], as_index=False)["pm25"].median()
    h24["hour_int"] = h24["timestamp_ist"].dt.hour
    h24["month_int"] = h24["timestamp_ist"].dt.month
    h24 = h24[h24["timestamp_ist"] < (train_clean["timestamp_ist"].min() - pd.Timedelta(days=20))].copy()

    ref_25 = train_clean.copy()
    ref_25["hour_int"] = ref_25["timestamp_ist"].dt.hour
    ref_25["month_int"] = ref_25["timestamp_ist"].dt.month

    ref_slice = ref_25[ref_25["month_int"].isin(months)].copy() if months else ref_25.copy()
    h24_slice = h24[h24["month_int"].isin(months)].copy() if months else h24.copy()

    def _iqr(series: pd.Series) -> float:
        q75, q25 = np.nanpercentile(series, [75, 25])
        return max(float(q75 - q25), 5.0)

    group_cols = ["location_id", "hour_int"] if align_method == "hour_med_iqr" else ["location_id"]
    g25 = ref_slice.groupby(group_cols)["pm25"].agg(med25="median", iqr25=_iqr).reset_index()
    g24 = h24_slice.groupby(group_cols)["pm25"].agg(med24="median", iqr24=_iqr).reset_index()

    h24_slice = h24_slice.merge(g24, on=group_cols, how="left").merge(g25, on=group_cols, how="left")
    scale_ratio = np.clip(h24_slice["iqr25"] / h24_slice["iqr24"].replace(0, np.nan), 0.60, 1.40).fillna(1.0)

    h24_slice["pm25_unaligned"] = h24_slice["pm25"]
    h24_slice["pm25"] = np.clip(
        h24_slice["med25"] + (h24_slice["pm25"] - h24_slice["med24"]) * scale_ratio,
        a_min=2.0,
        a_max=450.0,
    )

    if trace is not None:
        trace["hist_2024_calibration_table"] = h24_slice.copy()

    aligned = h24_slice[["location_id", "timestamp_ist", "pm25"]].sort_values("timestamp_ist").reset_index(drop=True)
    assert aligned["pm25"].isna().sum() == 0, "NaNs found in hist_2024_aligned!"
    assert (aligned["timestamp_ist"].dt.minute == 0).all(), "Non-hourly timestamps found in hist_2024_aligned!"

    print(
        f"[Validation Check 2D] Aligned 2024 History (months={months}, method={align_method}): "
        f"{len(aligned):,} hourly rows | All :00:00=True"
    )
    return aligned


def clean_all_datasets(
    raw_dir: Path,
    hist_2024_months: tuple[int, ...] | None = (10, 11, 12),
    align_method: str = "hour_med_iqr",
    save_parquet: bool = True,
    trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Orchestrates Block 1 & Block 2 cleaning and returns an ordered semantic stage ledger.
    Saves frozen Parquet checkpoints to data/processed/ when save_parquet=True.
    """
    train_raw, test_raw, sample_sub_raw, exact_station_map = load_raw_core_datasets(raw_dir)

    raw_aux = load_historical_archive(raw_dir, ["Other_Mumbai_Stations_2025_2026"])
    aux_clean = standardize_historical_df(raw_aux, "Other_Mumbai_Stations_2025_2026") if raw_aux is not None else None

    train_hw = apply_stage1_hardware_scrub(train_raw, trace=trace)
    train_clean = apply_stage2_spatial_scrub(train_hw, aux_clean=aux_clean, safe_aux=True, trace=trace)

    raw_hist_2024 = load_historical_archive(raw_dir, ["train_mumbai_2024_auxiliary", "History_Mumbai_2024"])
    hist_2024_aligned = (
        align_history_2024(
            raw_hist_2024=raw_hist_2024,
            train_clean=train_clean,
            exact_station_map=exact_station_map,
            months=hist_2024_months,
            align_method=align_method,
            trace=trace,
        )
        if raw_hist_2024 is not None
        else None
    )

    if save_parquet:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        train_clean.to_parquet(PROCESSED_DATA_DIR / "train_clean.parquet", index=False)
        if hist_2024_aligned is not None:
            hist_2024_aligned.to_parquet(PROCESSED_DATA_DIR / "hist_2024_aligned.parquet", index=False)
        print(f"[Validation Check 2E] Saved frozen Parquet checkpoints to {PROCESSED_DATA_DIR}")

    return {
        "train_raw": train_raw,
        "train_clean": train_clean,
        "test_raw": test_raw,
        "sample_sub_raw": sample_sub_raw,
        "exact_station_map": exact_station_map,
        "aux_clean": aux_clean,
        "hist_2024_aligned": hist_2024_aligned,
    }
