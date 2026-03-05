"""
resample_v2.py
Step 2 of 5 - BTCDataset v2

Resamples 1m raw parquet files into 5m, 15m, 30m, 1h, 4h, 1d.
Joins mark price, index price, and funding rate onto perp bars.

Input  (data/raw/):
  BTCUSDT_spot_1m_raw.parquet
  BTCUSDT_perp_1m_raw.parquet
  BTCUSDT_perp_mark_1m_raw.parquet
  BTCUSDT_perp_index_1m_raw.parquet
  BTCUSDT_perp_funding_raw.parquet

Output (data/resampled/):
  BTCUSDT_spot_{interval}.parquet   x6
  BTCUSDT_perp_{interval}.parquet   x6

Each bar includes:
  - OHLCV from 1m children
  - meta_is_complete, meta_observed/expected/missing_1m_count
  Perp additionally:
  - mark_open/high/low/close, index_open/close
  - basis_mark_pct, basis_last_pct
  - fund_rate_period, fund_rate_cum_24h, fund_rate_event_count

Usage:
  python scripts/resample_v2.py
  python scripts/resample_v2.py --instrument spot
  python scripts/resample_v2.py --instrument perp --intervals 1h 4h 1d

Requires:
  pip install pandas pyarrow
"""

import sys
import logging
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# ==============================================================================
# CONFIG
# ==============================================================================

BASE_DIR      = Path(r"C:\Users\tjall\Desktop\Trading\BTCDataset_v2")
RAW_DIR       = BASE_DIR / "data" / "raw"
RESAMPLED_DIR = BASE_DIR / "data" / "resampled"
LOG_DIR       = BASE_DIR / "logs"
SYMBOL        = "BTCUSDT"

INTERVALS = ["5m", "15m", "30m", "1h", "4h", "1d"]

EXPECTED_1M = {
    "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "4h": 240, "1d": 1440,
}

RESAMPLE_RULE = {
    "5m": "5min", "15m": "15min", "30m": "30min",
    "1h": "1h",   "4h": "4h",    "1d":  "1D",
}

# ==============================================================================
# LOGGING
# ==============================================================================

LOG_DIR.mkdir(parents=True, exist_ok=True)
RESAMPLED_DIR.mkdir(parents=True, exist_ok=True)

log_path = LOG_DIR / f"resample_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)

# ==============================================================================
# LOAD
# ==============================================================================

def load_1m(instrument):
    path = RAW_DIR / f"{SYMBOL}_{instrument}_1m_raw.parquet"
    if not path.exists():
        log.error(f"  File not found: {path}")
        return None
    log.info(f"  Loading {path.name}...")
    df = pd.read_parquet(path)
    df = df.sort_values("bar_start_ts_ms").reset_index(drop=True)
    idx = pd.to_datetime(df["bar_start_ts_ms"], unit="ms", utc=True)
    df.index = idx
    df.index.name = "bar_start_ts_utc"
    log.info(f"  {len(df):,} rows  {df.index.min()} -> {df.index.max()}")
    return df

def load_aux(fname, label):
    path = RAW_DIR / fname
    if not path.exists():
        log.warning(f"  {label} not found - skipping")
        return None
    df = pd.read_parquet(path)
    idx = pd.to_datetime(df["bar_start_ts_ms"], unit="ms", utc=True)
    df.index = idx
    df.index.name = "bar_start_ts_utc"
    return df.sort_index()

def load_funding():
    path = RAW_DIR / f"{SYMBOL}_perp_funding_raw.parquet"
    if not path.exists():
        log.warning("  Funding file not found - skipping")
        return None
    df = pd.read_parquet(path).sort_values("fund_time_utc").reset_index(drop=True)
    log.info(f"  Funding: {len(df):,} events  "
             f"{df['fund_time_utc'].min()} -> {df['fund_time_utc'].max()}")
    return df

# ==============================================================================
# RESAMPLE
# ==============================================================================

def resample_ohlcv(df_1m, rule):
    agg = df_1m.resample(rule, label="left", closed="left").agg(
        open                  =("open",            "first"),
        high                  =("high",            "max"),
        low                   =("low",             "min"),
        close                 =("close",           "last"),
        volume_base           =("volume_base",     "sum"),
        volume_quote          =("volume_quote",    "sum"),
        trade_count           =("trade_count",     "sum"),
        taker_buy_base        =("taker_buy_base",  "sum"),
        taker_buy_quote       =("taker_buy_quote", "sum"),
        meta_observed_1m_count=("open",            "count"),
    )
    return agg[agg["meta_observed_1m_count"] > 0].copy()

def add_completeness(df, interval):
    expected = EXPECTED_1M[interval]
    df["meta_expected_1m_count"] = expected
    df["meta_missing_1m_count"]  = (
        expected - df["meta_observed_1m_count"]
    ).clip(lower=0).astype(int)
    df["meta_is_complete"] = df["meta_observed_1m_count"] >= expected
    return df

def add_bar_times(df, interval):
    df = df.copy()
    # Index is still the DatetimeIndex from resample - insert as column
    df.insert(0, "bar_start_ts_utc", df.index)
    df = df.reset_index(drop=True)
    df["bar_start_ts_ms"] = (
        df["bar_start_ts_utc"].astype(np.int64) // 1_000_000
    )
    offset = pd.tseries.frequencies.to_offset(RESAMPLE_RULE[interval])
    df["bar_end_ts_utc"] = df["bar_start_ts_utc"] + offset
    df["bar_end_ts_ms"]  = (
        df["bar_end_ts_utc"].astype(np.int64) // 1_000_000
    )
    return df

def add_metadata(df, instrument, interval):
    df["meta_symbol"]          = SYMBOL
    df["meta_instrument_type"] = instrument
    df["meta_price_type"]      = "last"
    df["meta_interval"]        = interval
    df["meta_ohlc_valid"] = (
        (df["low"]  <= df["open"])  &
        (df["low"]  <= df["close"]) &
        (df["high"] >= df["open"])  &
        (df["high"] >= df["close"]) &
        (df["volume_base"] >= 0)
    )
    return df

# ==============================================================================
# MARK / INDEX JOIN
# ==============================================================================

def resample_aux(df_aux, rule, rename_map):
    if df_aux is None:
        return None
    agg = df_aux.resample(rule, label="left", closed="left").agg(
        **{new: (old, fn) for new, (old, fn) in rename_map.items()}
    )
    first_col = list(rename_map.keys())[0]
    return agg[agg[first_col].notna()].copy()

def join_mark_index(df, mark_agg, index_agg):
    df = df.set_index("bar_start_ts_utc")
    if mark_agg is not None:
        df = df.join(mark_agg, how="left")
    if index_agg is not None:
        df = df.join(index_agg, how="left")
    # Basis = (perp - index) / index * 100
    if "mark_close" in df.columns and "index_close" in df.columns:
        df["basis_mark_pct"] = (
            (df["mark_close"] - df["index_close"]) / df["index_close"] * 100
        )
    if "close" in df.columns and "index_close" in df.columns:
        df["basis_last_pct"] = (
            (df["close"] - df["index_close"]) / df["index_close"] * 100
        )
    return df.reset_index()

# ==============================================================================
# FUNDING JOIN
# ==============================================================================

def assign_funding(df, funding_df, interval):
    if funding_df is None or len(funding_df) == 0:
        df["fund_rate_period"]      = np.nan
        df["fund_rate_event_count"] = 0
        df["fund_rate_cum_24h"]     = np.nan
        return df

    fund_times = (
        funding_df["fund_time_utc"].astype(np.int64) // 1_000_000
    ).values
    fund_rates = funding_df["fund_rate"].values

    df = df.sort_values("bar_start_ts_ms").copy()
    starts = df["bar_start_ts_ms"].values
    ends   = df["bar_end_ts_ms"].values
    n      = len(starts)

    rate_sum = np.zeros(n, dtype=float)
    ev_count = np.zeros(n, dtype=int)

    for i in range(n):
        lo = np.searchsorted(fund_times, starts[i], side="left")
        hi = np.searchsorted(fund_times, ends[i],   side="left")
        if hi > lo:
            rate_sum[i] = fund_rates[lo:hi].sum()
            ev_count[i] = hi - lo

    df["fund_rate_period"]      = rate_sum
    df["fund_rate_event_count"] = ev_count

    # Rolling 24h cumulative
    bars_per_24h = max(1, 1440 // EXPECTED_1M.get(interval, 1))
    df["fund_rate_cum_24h"] = (
        df["fund_rate_period"].rolling(bars_per_24h, min_periods=1).sum()
    )
    return df.reset_index(drop=True)

# ==============================================================================
# FULL PIPELINE FOR ONE INTERVAL
# ==============================================================================

def resample_interval(df_1m, interval, instrument,
                      mark_1m=None, index_1m=None, funding_df=None):
    rule = RESAMPLE_RULE[interval]
    log.info(f"    {interval}...")

    agg = resample_ohlcv(df_1m, rule)
    agg = add_completeness(agg, interval)
    agg = add_bar_times(agg, interval)
    agg = add_metadata(agg, instrument, interval)

    if instrument == "perp":
        mark_agg = resample_aux(mark_1m, rule, {
            "mark_open":  ("open",  "first"),
            "mark_high":  ("high",  "max"),
            "mark_low":   ("low",   "min"),
            "mark_close": ("close", "last"),
        })
        index_agg = resample_aux(index_1m, rule, {
            "index_open":  ("open",  "first"),
            "index_close": ("close", "last"),
        })
        agg = join_mark_index(agg, mark_agg, index_agg)
        agg = assign_funding(agg, funding_df, interval)

    complete_pct = agg["meta_is_complete"].mean() * 100
    log.info(
        f"      {len(agg):,} bars | complete: {complete_pct:.1f}% | "
        f"cols: {len(agg.columns)}"
    )
    return agg

# ==============================================================================
# SAVE
# ==============================================================================

def save(df, instrument, interval):
    fname = f"{SYMBOL}_{instrument}_{interval}.parquet"
    path  = RESAMPLED_DIR / fname
    df.to_parquet(path, index=False, engine="pyarrow")
    mb = path.stat().st_size / 1024**2
    log.info(f"      Saved: {fname}  ({mb:.1f} MB)")

# ==============================================================================
# MAIN
# ==============================================================================

def run_instrument(instrument, intervals, mark_1m=None,
                   index_1m=None, funding_df=None):
    log.info(f"\n{'='*60}")
    log.info(f"  Resampling: {instrument.upper()}")
    log.info(f"  Intervals:  {intervals}")
    log.info(f"{'='*60}")

    df_1m = load_1m(instrument)
    if df_1m is None:
        log.error(f"  Cannot resample {instrument} - 1m file missing")
        return

    for interval in intervals:
        try:
            df = resample_interval(
                df_1m, interval, instrument,
                mark_1m    = mark_1m,
                index_1m   = index_1m,
                funding_df = funding_df,
            )
            save(df, instrument, interval)
        except Exception as e:
            import traceback
            log.error(f"  ERROR on {instrument} {interval}: {e}")
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instrument", choices=["spot","perp","both"],
                        default="both")
    parser.add_argument("--intervals", nargs="+", default=INTERVALS,
                        choices=INTERVALS)
    args = parser.parse_args()

    instruments = (["spot","perp"] if args.instrument == "both"
                   else [args.instrument])

    log.info("=" * 60)
    log.info("  BTCDataset v2 - Step 2: Resample")
    log.info(f"  Instruments: {instruments}")
    log.info(f"  Intervals:   {args.intervals}")
    log.info(f"  Output:      {RESAMPLED_DIR}")
    log.info("=" * 60)

    mark_1m = index_1m = funding_df = None
    if "perp" in instruments:
        log.info("\nLoading perp supplementary data...")
        mark_1m    = load_aux(f"{SYMBOL}_perp_mark_1m_raw.parquet",  "Mark price")
        index_1m   = load_aux(f"{SYMBOL}_perp_index_1m_raw.parquet", "Index price")
        funding_df = load_funding()

    for instrument in instruments:
        run_instrument(
            instrument, args.intervals,
            mark_1m    = mark_1m    if instrument == "perp" else None,
            index_1m   = index_1m   if instrument == "perp" else None,
            funding_df = funding_df if instrument == "perp" else None,
        )

    log.info("\n" + "=" * 60)
    log.info("  Step 2 complete.")
    log.info(f"  Files in {RESAMPLED_DIR}:")
    for f in sorted(RESAMPLED_DIR.glob("*.parquet")):
        mb = f.stat().st_size / 1024**2
        log.info(f"    {f.name:<45}  {mb:>8.1f} MB")
    log.info("\n  Next: run scripts/enrich_ict_v4.py")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
