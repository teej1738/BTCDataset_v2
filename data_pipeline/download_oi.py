"""
download_oi.py -- Download Binance OI metrics from data.binance.vision
D37a: OI data pipeline for BTCDataset_v2

Downloads daily 5m OI metrics (open interest, long/short ratios, taker volume
ratio) from the Binance public data archive. Verifies SHA256 checksums, computes
derived features, and saves as monthly parquet files.

URL pattern:
  https://data.binance.vision/data/futures/um/daily/metrics/BTCUSDT/
    BTCUSDT-metrics-YYYY-MM-DD.zip
  Checksum: same URL + .CHECKSUM

Output: core/data/raw/oi_metrics/oi_metrics_YYYY-MM.parquet (one per month)

Usage: python data_pipeline/download_oi.py
"""

import os
import io
import sys
import time
import hashlib
import zipfile
import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "core", "data", "raw", "oi_metrics")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
BASE_URL = (
    "https://data.binance.vision/data/futures/um/daily/metrics/BTCUSDT"
)

START_DATE = date(2020, 2, 11)
END_DATE = date.today() - timedelta(days=1)

BARS_PER_HOUR = 12       # 5m bars per hour
BARS_PER_4H = 48         # 5m bars per 4 hours
ZSCORE_WINDOW = 20        # rolling window for OI z-score

REQUEST_TIMEOUT = 20      # seconds
SLEEP_BETWEEN = 0.03      # seconds between day downloads
MAX_RETRIES = 2           # retries per day on transient errors

# Column rename map: Binance CSV -> project convention
RENAME = {
    "create_time": "bar_start_ts_utc",
    "sum_open_interest": "oi_btc",
    "sum_open_interest_value": "oi_usdt",
    "count_toptrader_long_short_ratio": "toptrader_ls_ratio_count",
    "sum_toptrader_long_short_ratio": "toptrader_ls_ratio_position",
    "count_long_short_ratio": "global_ls_ratio",
    "sum_taker_long_short_vol_ratio": "taker_ls_vol_ratio",
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "download_oi.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------
def download_day(session, dt):
    """Download and verify one day of metrics. Returns (DataFrame, None) or
    (None, error_string)."""
    ds = dt.strftime("%Y-%m-%d")
    zip_name = f"BTCUSDT-metrics-{ds}.zip"
    zip_url = f"{BASE_URL}/{zip_name}"
    ck_url = f"{zip_url}.CHECKSUM"

    # -- fetch checksum --
    try:
        r_ck = session.get(ck_url, timeout=REQUEST_TIMEOUT)
        if r_ck.status_code == 404:
            return None, "not available (404)"
        if r_ck.status_code != 200:
            return None, f"checksum HTTP {r_ck.status_code}"
        expected_hash = r_ck.text.strip().split()[0]
    except requests.RequestException as exc:
        return None, f"checksum request error: {exc}"

    # -- fetch zip --
    try:
        r_zip = session.get(zip_url, timeout=REQUEST_TIMEOUT)
        if r_zip.status_code == 404:
            return None, "zip not available (404)"
        if r_zip.status_code != 200:
            return None, f"zip HTTP {r_zip.status_code}"
    except requests.RequestException as exc:
        return None, f"zip request error: {exc}"

    # -- verify SHA-256 --
    actual_hash = hashlib.sha256(r_zip.content).hexdigest()
    if actual_hash != expected_hash:
        return None, (
            f"SHA256 mismatch: expected {expected_hash[:16]}... "
            f"got {actual_hash[:16]}..."
        )

    # -- parse CSV inside zip --
    try:
        zf = zipfile.ZipFile(io.BytesIO(r_zip.content))
        csv_name = zf.namelist()[0]
        with zf.open(csv_name) as f:
            df = pd.read_csv(f)
        if len(df.columns) < 8:
            return None, f"unexpected column count: {len(df.columns)}"
        return df, None
    except Exception as exc:
        return None, f"parse error: {exc}"


def download_day_with_retry(session, dt):
    """Wrapper with retry logic for transient errors."""
    for attempt in range(MAX_RETRIES + 1):
        df, err = download_day(session, dt)
        if df is not None:
            return df, None
        # Do not retry 404s or parse errors
        if err and ("404" in err or "parse" in err or "mismatch" in err):
            return None, err
        if attempt < MAX_RETRIES:
            time.sleep(1.0)  # back off before retry
    return None, err


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def compute_derived_features(df):
    """Compute OI-derived features in-place."""
    oi = df["oi_btc"]

    # Absolute OI change over 1h (12 bars) and 4h (48 bars)
    df["oi_change_1h"] = oi.diff(BARS_PER_HOUR)
    df["oi_change_4h"] = oi.diff(BARS_PER_4H)

    # Percentage OI change over 1h
    df["oi_change_pct_1h"] = oi.pct_change(BARS_PER_HOUR) * 100.0

    # Rolling 20-bar z-score of OI
    rm = oi.rolling(ZSCORE_WINDOW).mean()
    rs = oi.rolling(ZSCORE_WINDOW).std()
    df["oi_zscore_20"] = (oi - rm) / rs

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("download_oi.py -- Binance OI Metrics Download")
    print("=" * 60)
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Build date list
    dates = []
    dt = START_DATE
    while dt <= END_DATE:
        dates.append(dt)
        dt += timedelta(days=1)
    total_days = len(dates)

    print(f"Date range  : {START_DATE} to {END_DATE}")
    print(f"Days to fetch: {total_days}")
    print(f"Output dir  : {OUTPUT_DIR}")
    print()

    # -----------------------------------------------------------------------
    # Phase 1: Download
    # -----------------------------------------------------------------------
    print("Phase 1: Downloading daily zip files...")
    print()

    session = requests.Session()
    all_dfs = []
    missing = []
    t0 = time.time()

    for i, dt in enumerate(dates):
        df, err = download_day_with_retry(session, dt)
        if df is not None:
            all_dfs.append(df)
        else:
            missing.append((dt, err))
            log.warning("Missing %s: %s", dt, err)

        # Progress every 200 days and at end
        if (i + 1) % 200 == 0 or (i + 1) == total_days:
            elapsed = time.time() - t0
            pct = (i + 1) / total_days * 100
            print(
                f"  [{i+1:>5}/{total_days}] {pct:5.1f}%%  "
                f"ok={len(all_dfs)}  missing={len(missing)}  "
                f"elapsed={elapsed:.0f}s"
            )

        time.sleep(SLEEP_BETWEEN)

    session.close()
    elapsed_total = time.time() - t0

    print()
    print(f"Download complete in {elapsed_total:.0f}s")
    print(f"  Downloaded: {len(all_dfs)}/{total_days} days")
    print(f"  Missing   : {len(missing)}")

    if missing:
        print()
        print("Missing dates (first 30):")
        for dt_m, err_m in missing[:30]:
            print(f"  {dt_m}: {err_m}")
        if len(missing) > 30:
            print(f"  ... and {len(missing) - 30} more")

    if not all_dfs:
        print()
        print("ERROR: No data downloaded. Exiting.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Phase 2: Concatenate and process
    # -----------------------------------------------------------------------
    print()
    print("Phase 2: Processing...")

    df = pd.concat(all_dfs, ignore_index=True)
    del all_dfs  # free memory

    # Rename columns
    df = df.rename(columns=RENAME)
    df = df.drop(columns=["symbol"], errors="ignore")

    # Parse and localize timestamps
    df["bar_start_ts_utc"] = pd.to_datetime(
        df["bar_start_ts_utc"], utc=True
    )

    # Sort and deduplicate
    df = df.sort_values("bar_start_ts_utc").reset_index(drop=True)
    n_before = len(df)
    df = df.drop_duplicates(subset=["bar_start_ts_utc"], keep="first")
    n_dupes = n_before - len(df)
    if n_dupes:
        print(f"  Removed {n_dupes} duplicate timestamps")

    # Numeric coercion (safety)
    numeric_cols = [c for c in df.columns if c != "bar_start_ts_utc"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Compute derived features
    df = compute_derived_features(df)

    # Final column order
    final_cols = [
        "bar_start_ts_utc",
        "oi_btc",
        "oi_usdt",
        "toptrader_ls_ratio_count",
        "toptrader_ls_ratio_position",
        "global_ls_ratio",
        "taker_ls_vol_ratio",
        "oi_change_1h",
        "oi_change_4h",
        "oi_change_pct_1h",
        "oi_zscore_20",
    ]
    df = df[final_cols]

    # -----------------------------------------------------------------------
    # Phase 3: Save monthly parquet
    # -----------------------------------------------------------------------
    print()
    print("Phase 3: Saving monthly parquet files...")

    df["_ym"] = df["bar_start_ts_utc"].dt.to_period("M")
    months = sorted(df["_ym"].unique())
    saved = 0

    for ym in months:
        mdf = df[df["_ym"] == ym].drop(columns=["_ym"]).copy()
        fname = f"oi_metrics_{ym}.parquet"
        fpath = os.path.join(OUTPUT_DIR, fname)
        mdf.to_parquet(fpath, engine="pyarrow", compression="zstd", index=False)
        saved += 1

    df = df.drop(columns=["_ym"])
    print(f"  Saved {saved} monthly parquet files")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print()
    print(f"Total rows     : {len(df):,}")
    ts_min = df["bar_start_ts_utc"].min()
    ts_max = df["bar_start_ts_utc"].max()
    print(f"Date range     : {ts_min} to {ts_max}")
    print(f"Columns        : {len(df.columns)} ({', '.join(df.columns)})")
    print(f"Missing dates  : {len(missing)}")
    print(f"Monthly files  : {saved}")
    print(f"Output dir     : {OUTPUT_DIR}")
    print()

    # Sample output
    print("First 5 rows:")
    print("-" * 60)
    sample = df.head(5)
    for col in sample.columns:
        if col == "bar_start_ts_utc":
            vals = sample[col].astype(str).tolist()
        else:
            vals = [f"{v:.4f}" if pd.notna(v) else "NaN" for v in sample[col]]
        print(f"  {col:30s}: {vals}")

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)

    log.info(
        "Completed: %d rows, %d days downloaded, %d missing, %d months saved",
        len(df), total_days - len(missing), len(missing), saved,
    )


if __name__ == "__main__":
    main()
