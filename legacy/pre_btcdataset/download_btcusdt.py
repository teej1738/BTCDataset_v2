"""
Script 1 — download_btcusdt.py
Binance Futures BTCUSDT Historical Data Downloader
Pulls OHLCV + futures-specific data from Binance Futures API (fapi.binance.com)

Timeframes: 5m, 15m, 30m, 1h, 4h, 1d
Date range:  2017-08-17 to 2026-03-01

Requires: pip install requests pandas numpy pyarrow
"""

import requests
import pandas as pd
import numpy as np
import time
import os
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

SYMBOL     = "BTCUSDT"
INTERVALS  = ["5m", "15m", "30m", "1h", "4h", "1d"]
OUTPUT_DIR = r"C:\Users\tjall\Desktop\Trading\data"
START_DATE = "2017-08-17"
END_DATE   = "2026-03-01"
LIMIT      = 1000

FAPI_BASE  = "https://fapi.binance.com"
SPOT_BASE  = "https://api.binance.com"
ATR_PERIOD = 14
RVOL_PERIOD = 20

OI_INTERVAL_MAP = {
    "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1h", "4h":  "4h",  "1d":  "1d",
}

# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════

def setup_logging(output_dir):
    log_path = os.path.join(output_dir, "data_quality.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def date_to_ms(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)

def ms_to_dt(ms):
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)

INTERVAL_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
    "30m": 1_800_000, "1h": 3_600_000, "2h": 7_200_000,
    "4h": 14_400_000, "6h": 21_600_000, "8h": 28_800_000,
    "12h": 43_200_000, "1d": 86_400_000,
}

def get_checkpoint(checkpoint_file, key):
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            return json.load(f).get(key)
    return None

def save_checkpoint(checkpoint_file, key, value):
    data = {}
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            data = json.load(f)
    data[key] = value
    with open(checkpoint_file, "w") as f:
        json.dump(data, f, indent=2)

def safe_get(url, params, retries=5, logger=None):
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code == 429:
                wait = 60
                if logger: logger.warning(f"Rate limited. Waiting {wait}s...")
                time.sleep(wait)
                continue
            if resp.status_code in (400, 404):
                return []
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            wait = 2 ** attempt
            if logger: logger.warning(f"Request failed ({e}). Retry {attempt+1}/{retries} in {wait}s")
            time.sleep(wait)
    if logger: logger.error(f"All retries failed for {url}")
    return []

# ══════════════════════════════════════════════════════════════════════════════
# OHLCV
# ══════════════════════════════════════════════════════════════════════════════

KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_vol", "trades",
    "taker_base", "taker_quote", "ignore"
]

def fetch_ohlcv(symbol, interval, start_ms, end_ms, checkpoint_file, logger):
    url = f"{SPOT_BASE}/api/v3/klines"
    all_rows = []
    ck_key = f"ohlcv_{interval}"
    current_start = get_checkpoint(checkpoint_file, ck_key) or start_ms
    if current_start != start_ms:
        logger.info(f"  Resuming {interval} OHLCV from: {ms_to_dt(current_start)}")
    interval_ms = INTERVAL_MS[interval]

    while current_start < end_ms:
        params = {
            "symbol": symbol, "interval": interval,
            "startTime": current_start,
            "endTime": min(end_ms, current_start + LIMIT * interval_ms),
            "limit": LIMIT,
        }
        data = safe_get(url, params, logger=logger)
        if not data:
            break
        all_rows.extend(data)
        last_open_time = data[-1][0]
        current_start = last_open_time + interval_ms
        save_checkpoint(checkpoint_file, ck_key, current_start)
        logger.info(f"  OHLCV {interval} — up to {ms_to_dt(last_open_time).strftime('%Y-%m-%d %H:%M')} ({len(all_rows):,} candles)")
        time.sleep(0.1)
        # Stop only if we've reached end_ms (not just because page was smaller than LIMIT)
        if current_start >= end_ms:
            break

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows, columns=KLINE_COLS)
    df.drop(columns=["ignore"], inplace=True)
    for col in ["open", "high", "low", "close", "volume", "quote_vol", "taker_base", "taker_quote"]:
        df[col] = df[col].astype(float)
    df["trades"]    = df["trades"].astype(int)
    df["open_time"] = df["open_time"].astype(np.int64)
    df["close_time"]= df["close_time"].astype(np.int64)
    df["open_time_utc"]  = pd.to_datetime(df["open_time"],  unit="ms", utc=True)
    df["close_time_utc"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df = df.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)
    return df

# ══════════════════════════════════════════════════════════════════════════════
# FUTURES DATA
# ══════════════════════════════════════════════════════════════════════════════

def fetch_funding_rate(symbol, start_ms, end_ms, checkpoint_file, logger):
    url = f"{FAPI_BASE}/fapi/v1/fundingRate"
    all_rows = []
    ck_key = "funding_rate"
    current_start = get_checkpoint(checkpoint_file, ck_key) or start_ms
    if current_start != start_ms:
        logger.info(f"  Resuming funding rate from: {ms_to_dt(current_start)}")

    while current_start < end_ms:
        params = {"symbol": symbol, "startTime": current_start, "endTime": end_ms, "limit": 1000}
        data = safe_get(url, params, logger=logger)
        if not data:
            break
        all_rows.extend(data)
        last_time = data[-1]["fundingTime"]
        current_start = last_time + 1
        save_checkpoint(checkpoint_file, ck_key, current_start)
        logger.info(f"  Funding rate — up to {ms_to_dt(last_time).strftime('%Y-%m-%d %H:%M')} ({len(all_rows):,} records)")
        time.sleep(0.1)
        if len(data) < 1000:
            break

    if not all_rows:
        logger.warning("  No funding rate data retrieved.")
        return pd.DataFrame(columns=["funding_time_ms", "funding_rate"])

    df = pd.DataFrame(all_rows)
    df = df.rename(columns={"fundingTime": "funding_time_ms", "fundingRate": "funding_rate"})
    df["funding_rate"]    = df["funding_rate"].astype(float)
    df["funding_time_ms"] = df["funding_time_ms"].astype(np.int64)
    return df[["funding_time_ms", "funding_rate"]].sort_values("funding_time_ms").drop_duplicates("funding_time_ms").reset_index(drop=True)


def fetch_open_interest(symbol, interval, start_ms, end_ms, checkpoint_file, logger):
    url = f"{FAPI_BASE}/futures/data/openInterestHist"
    all_rows = []
    ck_key = f"oi_{interval}"
    current_start = get_checkpoint(checkpoint_file, ck_key) or start_ms
    itvl_ms = INTERVAL_MS.get(interval, 3_600_000)

    while current_start < end_ms:
        params = {
            "symbol": symbol, "period": interval,
            "startTime": current_start,
            "endTime": min(end_ms, current_start + 500 * itvl_ms),
            "limit": 500,
        }
        data = safe_get(url, params, logger=logger)
        if not data:
            break
        all_rows.extend(data)
        last_time = data[-1]["timestamp"]
        current_start = last_time + itvl_ms
        save_checkpoint(checkpoint_file, ck_key, current_start)
        logger.info(f"  Open interest {interval} — up to {ms_to_dt(last_time).strftime('%Y-%m-%d %H:%M')} ({len(all_rows):,} records)")
        time.sleep(0.1)
        if len(data) < 500:
            break

    if not all_rows:
        logger.warning(f"  No open interest data for {interval} (Binance history may be limited).")
        return pd.DataFrame(columns=["timestamp_ms", "open_interest", "oi_value"])

    df = pd.DataFrame(all_rows)
    df = df.rename(columns={"timestamp": "timestamp_ms", "sumOpenInterest": "open_interest", "sumOpenInterestValue": "oi_value"})
    df["open_interest"] = df["open_interest"].astype(float)
    df["oi_value"]      = df["oi_value"].astype(float)
    df["timestamp_ms"]  = df["timestamp_ms"].astype(np.int64)
    return df[["timestamp_ms", "open_interest", "oi_value"]].sort_values("timestamp_ms").drop_duplicates("timestamp_ms").reset_index(drop=True)


def fetch_long_short_ratio(symbol, interval, start_ms, end_ms, checkpoint_file, logger):
    url = f"{FAPI_BASE}/futures/data/globalLongShortAccountRatio"
    all_rows = []
    ck_key = f"ls_ratio_{interval}"
    current_start = get_checkpoint(checkpoint_file, ck_key) or start_ms
    itvl_ms = INTERVAL_MS.get(interval, 3_600_000)

    while current_start < end_ms:
        params = {
            "symbol": symbol, "period": interval,
            "startTime": current_start,
            "endTime": min(end_ms, current_start + 500 * itvl_ms),
            "limit": 500,
        }
        data = safe_get(url, params, logger=logger)
        if not data:
            break
        all_rows.extend(data)
        last_time = data[-1]["timestamp"]
        current_start = last_time + itvl_ms
        save_checkpoint(checkpoint_file, ck_key, current_start)
        logger.info(f"  L/S ratio {interval} — up to {ms_to_dt(last_time).strftime('%Y-%m-%d %H:%M')} ({len(all_rows):,} records)")
        time.sleep(0.1)
        if len(data) < 500:
            break

    if not all_rows:
        logger.warning(f"  No long/short ratio data for {interval} (Binance history is limited).")
        return pd.DataFrame(columns=["timestamp_ms", "long_account_ratio", "short_account_ratio", "ls_ratio"])

    df = pd.DataFrame(all_rows)
    df = df.rename(columns={"timestamp": "timestamp_ms", "longAccount": "long_account_ratio", "shortAccount": "short_account_ratio", "longShortRatio": "ls_ratio"})
    for col in ["long_account_ratio", "short_account_ratio", "ls_ratio"]:
        df[col] = df[col].astype(float)
    df["timestamp_ms"] = df["timestamp_ms"].astype(np.int64)
    return df[["timestamp_ms", "long_account_ratio", "short_account_ratio", "ls_ratio"]].sort_values("timestamp_ms").drop_duplicates("timestamp_ms").reset_index(drop=True)

# ══════════════════════════════════════════════════════════════════════════════
# DERIVED FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def add_volume_features(df):
    df["buy_vol"]  = df["taker_base"].astype(float)
    df["sell_vol"] = df["volume"] - df["buy_vol"]
    df["delta"]    = df["buy_vol"] - df["sell_vol"]
    df["_date"]    = df["open_time_utc"].dt.date
    df["cvd"]      = df.groupby("_date")["delta"].cumsum()
    df.drop(columns=["_date"], inplace=True)
    return df

def add_atr(df, period=ATR_PERIOD):
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    df[f"atr_{period}"] = tr.ewm(alpha=1/period, min_periods=period).mean()
    return df

def add_realized_vol(df, period=RVOL_PERIOD):
    log_ret = np.log(df["close"] / df["close"].shift(1))
    df[f"realized_vol_{period}"] = log_ret.rolling(period).std()
    return df

def add_session_labels(df):
    utc_mins = df["open_time_utc"].dt.hour * 60 + df["open_time_utc"].dt.minute
    conditions = [
        (utc_mins >= 60)  & (utc_mins < 300),   # 01:00-05:00 UTC = Asia killzone
        (utc_mins >= 420) & (utc_mins < 600),    # 07:00-10:00 UTC = London killzone
        (utc_mins >= 720) & (utc_mins < 900),    # 12:00-15:00 UTC = NY killzone
    ]
    df["session"] = np.select(conditions, ["Asia", "London", "NewYork"], default="Off")
    return df

def add_reference_prices(df):
    df = df.copy()
    dt = df["open_time_utc"]
    df["_date"]     = dt.dt.date
    df["_utc_mins"] = dt.dt.hour * 60 + dt.dt.minute

    # Midnight open (00:00 UTC)
    df["midnight_open"] = np.where(df["_utc_mins"] == 0, df["open"], np.nan)
    df["midnight_open"] = df.groupby("_date")["midnight_open"].transform("ffill")

    # 08:30 EST = 13:30 UTC = 810 mins
    df["ny_open_830"] = np.where(df["_utc_mins"] == 810, df["open"], np.nan)
    df["ny_open_830"] = df.groupby("_date")["ny_open_830"].transform("ffill")

    df.drop(columns=["_date", "_utc_mins"], inplace=True)
    return df

def add_previous_levels(df):
    df = df.copy()
    dt = df["open_time_utc"]
    df["_date"]  = dt.dt.date
    df["_week"]  = dt.dt.to_period("W")
    df["_month"] = dt.dt.to_period("M")

    # Previous Day
    daily = df.groupby("_date").agg(dh=("high","max"), dl=("low","min"), dc=("close","last")).reset_index()
    daily["pdh"] = daily["dh"].shift(1)
    daily["pdl"] = daily["dl"].shift(1)
    daily["pdc"] = daily["dc"].shift(1)
    df = df.merge(daily[["_date","pdh","pdl","pdc"]], on="_date", how="left")

    # Previous Week
    weekly = df.groupby("_week").agg(wh=("high","max"), wl=("low","min"), wc=("close","last")).reset_index()
    weekly["pwh"] = weekly["wh"].shift(1)
    weekly["pwl"] = weekly["wl"].shift(1)
    weekly["pwc"] = weekly["wc"].shift(1)
    df = df.merge(weekly[["_week","pwh","pwl","pwc"]], on="_week", how="left")

    # Previous Month
    monthly = df.groupby("_month").agg(mh=("high","max"), ml=("low","min"), mc=("close","last")).reset_index()
    monthly["pmh"] = monthly["mh"].shift(1)
    monthly["pml"] = monthly["ml"].shift(1)
    monthly["pmc"] = monthly["mc"].shift(1)
    df = df.merge(monthly[["_month","pmh","pml","pmc"]], on="_month", how="left")

    df.drop(columns=["_date","_week","_month"], inplace=True)
    return df

# ══════════════════════════════════════════════════════════════════════════════
# MERGING FUTURES DATA
# ══════════════════════════════════════════════════════════════════════════════

def merge_funding_rate(df, funding_df):
    if funding_df.empty:
        df["funding_rate"] = np.nan
        return df
    merged = pd.merge_asof(
        df.sort_values("open_time"),
        funding_df.rename(columns={"funding_time_ms": "open_time"}),
        on="open_time", direction="backward"
    )
    return merged.sort_values("open_time").reset_index(drop=True)

def merge_open_interest(df, oi_df):
    if oi_df.empty:
        df["open_interest"] = np.nan
        df["oi_value"]      = np.nan
        return df
    merged = pd.merge_asof(
        df.sort_values("open_time"),
        oi_df.rename(columns={"timestamp_ms": "open_time"}),
        on="open_time", direction="backward"
    )
    return merged.sort_values("open_time").reset_index(drop=True)

def merge_long_short(df, ls_df):
    if ls_df.empty:
        df["long_account_ratio"]  = np.nan
        df["short_account_ratio"] = np.nan
        df["ls_ratio"]            = np.nan
        return df
    merged = pd.merge_asof(
        df.sort_values("open_time"),
        ls_df.rename(columns={"timestamp_ms": "open_time"}),
        on="open_time", direction="backward"
    )
    return merged.sort_values("open_time").reset_index(drop=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATA QUALITY
# ══════════════════════════════════════════════════════════════════════════════

def validate_data(df, interval, logger):
    issues = []
    interval_ms = INTERVAL_MS[interval]

    dupes = df.duplicated("open_time").sum()
    if dupes > 0:
        msg = f"  WARNING: {dupes} duplicate candles"
        issues.append(msg); logger.warning(msg)

    times = df["open_time"].sort_values().values
    diffs = np.diff(times)
    gaps  = np.where(diffs > interval_ms * 1.5)[0]
    for g in gaps[:10]:
        missing = int(diffs[g] / interval_ms) - 1
        msg = f"  WARNING: {missing} missing candles between {ms_to_dt(int(times[g]))} and {ms_to_dt(int(times[g+1]))}"
        issues.append(msg); logger.warning(msg)
    if len(gaps) > 10:
        logger.warning(f"  WARNING: ...and {len(gaps)-10} more gaps (see log)")

    bad_ohlc = ((df["high"] < df["low"]) | (df["high"] < df["open"]) |
                (df["high"] < df["close"]) | (df["low"] > df["open"]) |
                (df["low"] > df["close"])).sum()
    if bad_ohlc > 0:
        msg = f"  WARNING: {bad_ohlc} candles with invalid OHLC"
        issues.append(msg); logger.warning(msg)

    zero_vol = (df["volume"] == 0).sum()
    if zero_vol > 0:
        msg = f"  WARNING: {zero_vol} zero-volume candles"
        issues.append(msg); logger.warning(msg)

    return {
        "total_candles": len(df), "duplicates": int(dupes),
        "gaps_found": int(len(gaps)), "bad_ohlc": int(bad_ohlc),
        "zero_vol_candles": int(zero_vol), "issues": issues,
    }

# ══════════════════════════════════════════════════════════════════════════════
# METADATA
# ══════════════════════════════════════════════════════════════════════════════

def write_metadata(output_dir, interval, df, quality, filename):
    meta = {
        "symbol": SYMBOL, "interval": interval, "filename": filename,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "start": str(df["open_time_utc"].min()),
        "end":   str(df["open_time_utc"].max()),
        "total_rows": len(df),
        "columns": list(df.columns),
        "data_quality": quality,
    }
    path = os.path.join(output_dir, f"{SYMBOL}_BINANCE_{interval}_metadata.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)

# ══════════════════════════════════════════════════════════════════════════════
# COLUMN ORDER
# ══════════════════════════════════════════════════════════════════════════════

FINAL_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_vol", "trades", "taker_base", "taker_quote",
    "open_time_utc", "close_time_utc",
    "buy_vol", "sell_vol", "delta", "cvd",
    f"atr_{ATR_PERIOD}", f"realized_vol_{RVOL_PERIOD}",
    "funding_rate", "open_interest", "oi_value",
    "long_account_ratio", "short_account_ratio", "ls_ratio",
    "session", "midnight_open", "ny_open_830",
    "pdh", "pdl", "pdc",
    "pwh", "pwl", "pwc",
    "pmh", "pml", "pmc",
]

def reorder_columns(df):
    existing = [c for c in FINAL_COLUMNS if c in df.columns]
    extra    = [c for c in df.columns if c not in existing]
    return df[existing + extra]

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def process_interval(interval, start_ms, end_ms, output_dir, logger):
    print(f"\n{'='*60}")
    print(f"  {SYMBOL} | {interval}")
    print(f"{'='*60}")

    checkpoint_file = os.path.join(output_dir, f".checkpoint_{interval}.json")

    logger.info(f"[{interval}] Fetching OHLCV...")
    df = fetch_ohlcv(SYMBOL, interval, start_ms, end_ms, checkpoint_file, logger)
    if df.empty:
        logger.error(f"[{interval}] No OHLCV data. Skipping.")
        return

    logger.info(f"[{interval}] Fetching funding rate...")
    funding_df = fetch_funding_rate(SYMBOL, start_ms, end_ms, checkpoint_file, logger)
    df = merge_funding_rate(df, funding_df)

    oi_interval = OI_INTERVAL_MAP.get(interval)
    if oi_interval:
        logger.info(f"[{interval}] Fetching open interest...")
        oi_df = fetch_open_interest(SYMBOL, oi_interval, start_ms, end_ms, checkpoint_file, logger)
        df = merge_open_interest(df, oi_df)
    else:
        df["open_interest"] = np.nan
        df["oi_value"]      = np.nan

    logger.info(f"[{interval}] Fetching long/short ratio...")
    ls_df = fetch_long_short_ratio(SYMBOL, interval, start_ms, end_ms, checkpoint_file, logger)
    df = merge_long_short(df, ls_df)

    logger.info(f"[{interval}] Computing derived features...")
    df = add_volume_features(df)
    df = add_atr(df)
    df = add_realized_vol(df)
    df = add_session_labels(df)
    df = add_reference_prices(df)
    df = add_previous_levels(df)

    logger.info(f"[{interval}] Validating data quality...")
    quality = validate_data(df, interval, logger)

    df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    df = reorder_columns(df)

    filename = f"{SYMBOL}_BINANCE_{interval}_{START_DATE}_to_{END_DATE}.csv"
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)

    write_metadata(output_dir, interval, df, quality, filename)

    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    logger.info(f"[{interval}] Saved {len(df):,} rows -> {filename}")
    print(f"\n  Done: {len(df):,} candles | {df['open_time_utc'].min()} -> {df['open_time_utc'].max()}")
    print(f"  Gaps: {quality['gaps_found']} | Bad OHLC: {quality['bad_ohlc']} | Zero vol: {quality['zero_vol_candles']}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger = setup_logging(OUTPUT_DIR)

    start_ms = date_to_ms(START_DATE)
    end_ms   = date_to_ms(END_DATE)

    logger.info(f"Starting: {SYMBOL} | {START_DATE} to {END_DATE}")
    logger.info(f"Intervals: {INTERVALS}")
    logger.info(f"Output: {OUTPUT_DIR}")

    for interval in INTERVALS:
        try:
            process_interval(interval, start_ms, end_ms, OUTPUT_DIR, logger)
        except Exception as e:
            logger.error(f"[{interval}] Error: {e}", exc_info=True)
            print(f"\n  FAILED: {interval} — {e}")
            print(f"  Re-run the script to resume from checkpoint.")

    print(f"\n{'='*60}")
    print(f"  All done. Files in: {OUTPUT_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
