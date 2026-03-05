"""
download_liquidations.py -- Download liquidation data for BTCUSDT perpetual
D37c: Liquidation data pipeline for BTCDataset_v2

Source: Coinalyze API v1/liquidation-history (daily granularity)
  - Symbol: BTCUSDT_PERP.A (Binance USDT-margined perpetual)
  - Full history from 2020-01-25 to present (daily data never deleted)
  - Values in BTC (base asset). convert_to_usd only works from 2022-01-22.
    USD conversion deferred to merge step (D37d) using close price.

Note on Binance data.binance.vision:
  The um (USDT-margined) liquidationSnapshot path does not exist --
  returns 404 for all dates and symbols. Only cm (coin-margined)
  BTCUSD_PERP data exists (2023-06 to 2024-10), which is a different
  instrument. Coinalyze daily is the only viable source for BTCUSDT_PERP
  liquidation history.

Note on granularity:
  Coinalyze retains intraday data on a rolling basis only (~7 days for
  5min, ~83 days for 1h). Historical intraday data cannot be backfilled.
  Daily granularity is the finest resolution available for full history.
  When merged to 5m bars (D37d), daily values will be forward-filled --
  all 288 bars in a day get the same liquidation values.

Output: core/data/raw/liquidations/liq_daily_YYYY-MM.parquet (one per month)

Usage: python data_pipeline/download_liquidations.py
"""

import os
import sys
import time
import logging
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "core", "data", "raw", "liquidations")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
ENV_FILE = os.path.join(PROJECT_ROOT, ".env")

API_BASE = "https://api.coinalyze.net/v1/liquidation-history"
SYMBOL = "BTCUSDT_PERP.A"
INTERVAL = "daily"

# Coinalyze daily data starts 2020-01-25 for BTCUSDT_PERP.A
START_DATE = date(2020, 1, 25)
END_DATE = date.today() - timedelta(days=1)

# API constraints
MAX_POINTS_PER_REQUEST = 1400   # stay under 1500 limit
RATE_LIMIT_SLEEP = 1.6          # seconds between requests (40 req/min)

# Derived feature parameters
CASCADE_LOOKBACK_DAYS = 7       # rolling mean window for cascade detection
CASCADE_MULTIPLIER = 3.0        # cascade threshold = multiplier x rolling mean
ZSCORE_WINDOW = 7               # rolling z-score window (days)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "download_liquidations.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_api_key():
    """Load COINALYZE_API_KEY from .env file."""
    if not os.path.exists(ENV_FILE):
        print(f"ERROR: .env not found at {ENV_FILE}")
        sys.exit(1)
    with open(ENV_FILE) as f:
        for line in f:
            line = line.strip()
            if line.startswith("COINALYZE_API_KEY="):
                key = line.split("=", 1)[1].strip()
                if not key:
                    print("ERROR: COINALYZE_API_KEY is empty in .env")
                    sys.exit(1)
                return key
    print("ERROR: COINALYZE_API_KEY not found in .env")
    sys.exit(1)


def date_to_ts(d):
    """Convert date to UTC unix timestamp (seconds)."""
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp())


def fetch_chunk(session, api_key, from_ts, to_ts):
    """Fetch one chunk of daily liquidation data from Coinalyze.
    Returns list of dicts or None on error."""
    params = {
        "symbols": SYMBOL,
        "interval": INTERVAL,
        "from": from_ts,
        "to": to_ts,
        "api_key": api_key,
    }
    try:
        r = session.get(API_BASE, params=params, timeout=20)
        if r.status_code == 429:
            # Rate limited -- wait and retry once
            retry_after = int(r.headers.get("Retry-After", 5))
            log.warning("Rate limited, waiting %ds", retry_after)
            print(f"  Rate limited, waiting {retry_after}s...")
            time.sleep(retry_after + 1)
            r = session.get(API_BASE, params=params, timeout=20)
        if r.status_code != 200:
            log.error("API error %d: %s", r.status_code, r.text[:200])
            return None
        data = r.json()
        if not data or not data[0].get("history"):
            return []
        return data[0]["history"]
    except Exception as exc:
        log.error("Request error: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("download_liquidations.py -- Coinalyze Liquidation Data")
    print("=" * 60)
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    api_key = load_api_key()
    print(f"API key loaded [{len(api_key)} chars]")

    # Build chunk boundaries (1400 days per request)
    chunk_days = MAX_POINTS_PER_REQUEST
    chunks = []
    d = START_DATE
    while d <= END_DATE:
        d_end = min(d + timedelta(days=chunk_days - 1), END_DATE)
        chunks.append((d, d_end))
        d = d_end + timedelta(days=1)

    print(f"Date range   : {START_DATE} to {END_DATE}")
    print(f"API chunks   : {len(chunks)}")
    print(f"Output dir   : {OUTPUT_DIR}")
    print()

    # -----------------------------------------------------------------------
    # Phase 1: Download
    # -----------------------------------------------------------------------
    print("Phase 1: Downloading daily liquidation data from Coinalyze...")
    print()

    session = requests.Session()
    all_records = []
    t0 = time.time()

    for i, (d_start, d_end) in enumerate(chunks):
        from_ts = date_to_ts(d_start)
        to_ts = date_to_ts(d_end) + 86399  # include full last day

        records = fetch_chunk(session, api_key, from_ts, to_ts)
        if records is None:
            print(f"  ERROR: chunk {i+1} ({d_start} to {d_end}) failed")
            log.error("Chunk %d failed: %s to %s", i + 1, d_start, d_end)
        elif len(records) == 0:
            print(f"  Chunk {i+1}/{len(chunks)}: {d_start} to {d_end} -- 0 points")
        else:
            all_records.extend(records)
            print(
                f"  Chunk {i+1}/{len(chunks)}: {d_start} to {d_end} "
                f"-- {len(records)} points (total: {len(all_records)})"
            )

        if i < len(chunks) - 1:
            time.sleep(RATE_LIMIT_SLEEP)

    session.close()
    elapsed = time.time() - t0

    print()
    print(f"Download complete in {elapsed:.0f}s")
    print(f"  Total daily points: {len(all_records)}")

    if not all_records:
        print("ERROR: No data downloaded. Exiting.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Phase 2: Process
    # -----------------------------------------------------------------------
    print()
    print("Phase 2: Processing...")

    # Build DataFrame
    df = pd.DataFrame(all_records)
    # Columns: t (unix seconds), l (longs liq USD), s (shorts liq USD)

    # Convert timestamp to datetime
    df["bar_start_ts_utc"] = pd.to_datetime(df["t"], unit="s", utc=True)
    df = df.drop(columns=["t"])

    # Rename -- values are in BTC (base asset)
    df = df.rename(columns={"l": "liq_long_btc", "s": "liq_short_btc"})

    # Sort and deduplicate
    df = df.sort_values("bar_start_ts_utc").reset_index(drop=True)
    n_before = len(df)
    df = df.drop_duplicates(subset=["bar_start_ts_utc"], keep="first")
    n_dupes = n_before - len(df)
    if n_dupes:
        print(f"  Removed {n_dupes} duplicate timestamps")

    # Compute derived features
    print("  Computing derived features...")

    # Total liquidation volume
    df["liq_total_btc"] = df["liq_long_btc"] + df["liq_short_btc"]

    # Ratio: fraction of liquidations that are longs
    df["liq_ratio"] = df["liq_long_btc"] / (df["liq_total_btc"] + 1e-9)

    # Cascade flag: liq_total > 3x 7-day rolling mean
    rolling_mean = df["liq_total_btc"].rolling(
        CASCADE_LOOKBACK_DAYS, min_periods=1
    ).mean()
    df["liq_cascade_flag"] = (
        df["liq_total_btc"] > CASCADE_MULTIPLIER * rolling_mean
    ).astype(np.int8)

    # Z-score of daily liq total over 7-day window
    rm = df["liq_total_btc"].rolling(ZSCORE_WINDOW, min_periods=2).mean()
    rs = df["liq_total_btc"].rolling(ZSCORE_WINDOW, min_periods=2).std()
    df["liq_zscore_7d"] = (df["liq_total_btc"] - rm) / rs

    # Change in liq total (1-day)
    df["liq_change_1d"] = df["liq_total_btc"].diff(1)

    # Change in liq total (percentage, 1-day)
    df["liq_change_pct_1d"] = df["liq_total_btc"].pct_change(1) * 100.0

    # Final column order
    final_cols = [
        "bar_start_ts_utc",
        "liq_long_btc",
        "liq_short_btc",
        "liq_total_btc",
        "liq_ratio",
        "liq_cascade_flag",
        "liq_zscore_7d",
        "liq_change_1d",
        "liq_change_pct_1d",
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
        fname = f"liq_daily_{ym}.parquet"
        fpath = os.path.join(OUTPUT_DIR, fname)
        mdf.to_parquet(
            fpath, engine="pyarrow", compression="zstd", index=False
        )
        saved += 1

    df = df.drop(columns=["_ym"])
    print(f"  Saved {saved} monthly parquet files")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    ts_min = df["bar_start_ts_utc"].min()
    ts_max = df["bar_start_ts_utc"].max()
    n_cascade = df["liq_cascade_flag"].sum()

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print()
    print(f"Source         : Coinalyze API (BTCUSDT_PERP.A, daily, BTC)")
    print(f"Total rows     : {len(df):,}")
    print(f"Date range     : {ts_min} to {ts_max}")
    print(f"Columns        : {len(df.columns)}")
    print(f"Monthly files  : {saved}")
    print(f"Output dir     : {OUTPUT_DIR}")
    print()
    print(f"Cascade events : {n_cascade} "
          f"({n_cascade/len(df)*100:.1f}% of days)")
    print()

    # Binance um note
    print("Binance source : SKIPPED")
    print("  Binance um (USDT-margined) liquidationSnapshot returns 404")
    print("  for all dates -- data does not exist on data.binance.vision.")
    print("  Only cm (coin-margined) BTCUSD_PERP exists (2023-06 to")
    print("  2024-10), which is a different instrument.")
    print()

    # Post-April-2021 NaN analysis
    apr_2021 = pd.Timestamp("2021-04-01", tz="UTC")
    post_apr = df[df["bar_start_ts_utc"] >= apr_2021]
    nan_rate = post_apr.isna().mean()
    print(f"Post-April-2021 coverage: {len(post_apr)} days")
    print(f"  NaN rate per column:")
    for col in df.columns:
        if col == "bar_start_ts_utc":
            continue
        nr = nan_rate[col]
        print(f"    {col:25s}: {nr:.4f} ({nr*100:.2f}%)")
    print()

    # Sample rows
    print("First 5 rows:")
    print("-" * 60)
    sample = df.head(5)
    for col in sample.columns:
        if col == "bar_start_ts_utc":
            vals = sample[col].astype(str).tolist()
        else:
            vals = [
                f"{v:.2f}" if pd.notna(v) else "NaN" for v in sample[col]
            ]
        print(f"  {col:25s}: {vals}")
    print()

    # Show cascade events
    cascades = df[df["liq_cascade_flag"] == 1]
    if len(cascades) > 0:
        print(f"Sample cascade events (up to 5):")
        print("-" * 60)
        sample_c = cascades.head(5)
        for _, row in sample_c.iterrows():
            ts = str(row["bar_start_ts_utc"])[:10]
            print(
                f"  {ts}  total={row['liq_total_btc']:,.2f} BTC  "
                f"long={row['liq_long_btc']:,.2f}  "
                f"short={row['liq_short_btc']:,.2f}  "
                f"zscore={row['liq_zscore_7d']:.2f}"
            )
    else:
        print("No cascade events found.")
    print()

    print("=" * 60)
    print("DONE")
    print("=" * 60)

    log.info(
        "Completed: %d rows, %s to %s, %d cascade events, %d months saved",
        len(df), ts_min, ts_max, n_cascade, saved,
    )


if __name__ == "__main__":
    main()
