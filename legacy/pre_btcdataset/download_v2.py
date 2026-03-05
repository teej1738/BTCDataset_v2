"""
download_v2.py
Step 1 of 5 - BTCDataset v2

Downloads BTCUSDT raw 1m data from Binance public archives.
No API key required. Uses official data.binance.vision archive.

Downloads:
  1. Spot 1m klines         (2017-08-01 onwards)
  2. Perp 1m klines         (2019-09-01 onwards)
  3. Perp mark price 1m     (2019-09-01 onwards)
  4. Perp index price 1m    (2019-09-01 onwards)
  5. Perp funding rate      (2019-09-01 onwards)

Output directory: BTCDataset_v2/data/raw/

Requires:
  pip install pandas pyarrow requests tqdm python-dateutil
"""

import os
import io
import sys
import time
import hashlib
import zipfile
import logging
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

# ==============================================================================
# CONFIG
# ==============================================================================

BASE_DIR   = Path(r"C:\Users\tjall\Desktop\Trading\BTCDataset_v2")
RAW_DIR    = BASE_DIR / "data" / "raw"
LOG_DIR    = BASE_DIR / "logs"
SYMBOL     = "BTCUSDT"

SPOT_START = datetime(2017, 8, 1, tzinfo=timezone.utc)
PERP_START = datetime(2019, 9, 1, tzinfo=timezone.utc)
END_DATE   = datetime(2026, 2, 1, tzinfo=timezone.utc)

ARCHIVE_BASE  = "https://data.binance.vision"
CHUNK_SIZE    = 1024 * 1024
MAX_RETRIES   = 3
RETRY_DELAY   = 5
REQUEST_DELAY = 0.3

# ==============================================================================
# LOGGING
# ==============================================================================

LOG_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

log_path = LOG_DIR / f"download_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
# URL BUILDERS
# ==============================================================================

def spot_urls(symbol, interval, year, month):
    fname = f"{symbol}-{interval}-{year}-{month:02d}.zip"
    base  = f"{ARCHIVE_BASE}/data/spot/monthly/klines/{symbol}/{interval}"
    return f"{base}/{fname}", f"{base}/{fname}.CHECKSUM"

def perp_urls(symbol, interval, year, month, price_type="klines"):
    fname = f"{symbol}-{interval}-{year}-{month:02d}.zip"
    base  = f"{ARCHIVE_BASE}/data/futures/um/monthly/{price_type}/{symbol}/{interval}"
    return f"{base}/{fname}", f"{base}/{fname}.CHECKSUM"

def funding_urls(symbol, year, month):
    fname = f"{symbol}-fundingRate-{year}-{month:02d}.zip"
    base  = f"{ARCHIVE_BASE}/data/futures/um/monthly/fundingRate/{symbol}"
    return f"{base}/{fname}", f"{base}/{fname}.CHECKSUM"

# ==============================================================================
# DOWNLOAD HELPERS
# ==============================================================================

def download_bytes(url):
    """Download URL to memory. Returns bytes or None if 404/error."""
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.content
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                log.warning(f"    Retry {attempt+1}: {e}")
                time.sleep(RETRY_DELAY)
            else:
                log.error(f"    Failed: {url} - {e}")
                return None

def check_sha256(data_bytes, checksum_bytes):
    """Returns True if SHA256 matches or checksum unavailable."""
    try:
        expected = checksum_bytes.decode("utf-8").strip().split()[0].lower()
        actual   = hashlib.sha256(data_bytes).hexdigest().lower()
        return actual == expected
    except Exception:
        return True

def unzip_csv(zip_bytes):
    """Extract first CSV from zip archive."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_names:
            raise ValueError(f"No CSV in zip. Files: {zf.namelist()}")
        return zf.read(csv_names[0])

# ==============================================================================
# PARSERS
# ==============================================================================

KLINE_COLS = [
    "bar_start_ts_ms", "open", "high", "low", "close",
    "volume_base", "bar_end_ts_ms", "volume_quote",
    "trade_count", "taker_buy_base", "taker_buy_quote", "ignore_field",
]

def parse_kline_csv(csv_bytes, instrument_type, price_type):
    df = pd.read_csv(io.BytesIO(csv_bytes), header=None,
                     names=KLINE_COLS, dtype=str)

    # Drop any header rows that snuck in
    df = df[pd.to_numeric(df["bar_start_ts_ms"], errors="coerce").notna()].copy()
    df = df.reset_index(drop=True)

    # Cast numerics
    for col in ["bar_start_ts_ms", "bar_end_ts_ms"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in ["open","high","low","close","volume_base","volume_quote",
                "taker_buy_base","taker_buy_quote"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    df["trade_count"] = pd.to_numeric(df["trade_count"], errors="coerce").astype("Int64")

    # UTC timestamps
    df["bar_start_ts_utc"] = pd.to_datetime(df["bar_start_ts_ms"], unit="ms", utc=True)
    df["bar_end_ts_utc"]   = pd.to_datetime(df["bar_end_ts_ms"],   unit="ms", utc=True)

    # Metadata
    df["meta_symbol"]          = SYMBOL
    df["meta_instrument_type"] = instrument_type
    df["meta_price_type"]      = price_type
    df["meta_interval"]        = "1m"

    # OHLC validity flag
    df["meta_ohlc_valid"] = (
        (df["low"]  <= df["open"])  &
        (df["low"]  <= df["close"]) &
        (df["high"] >= df["open"])  &
        (df["high"] >= df["close"]) &
        (df["volume_base"] >= 0)
    )

    df = df.drop(columns=["ignore_field"], errors="ignore")
    return df


FUNDING_COLS = ["fund_calc_time_ms", "fund_rate", "fund_mark_price"]

def parse_funding_csv(csv_bytes):
    df = pd.read_csv(io.BytesIO(csv_bytes), header=None,
                     names=FUNDING_COLS, dtype=str)

    # Drop header if present
    df = df[pd.to_numeric(df["fund_calc_time_ms"], errors="coerce").notna()].copy()
    df = df.reset_index(drop=True)

    df["fund_calc_time_ms"] = pd.to_numeric(df["fund_calc_time_ms"]).astype(np.int64)
    df["fund_rate"]         = pd.to_numeric(df["fund_rate"],         errors="coerce").astype(float)
    df["fund_mark_price"]   = pd.to_numeric(df["fund_mark_price"],   errors="coerce").astype(float)

    df["fund_time_utc"]        = pd.to_datetime(df["fund_calc_time_ms"], unit="ms", utc=True)
    df["meta_symbol"]          = SYMBOL
    df["meta_instrument_type"] = "perp"
    return df

# ==============================================================================
# GENERIC MONTHLY DOWNLOADER
# ==============================================================================

def download_monthly(name, url_fn, parse_fn, start, end):
    """
    Download and parse all monthly archive files for a series.
    url_fn(year, month) -> (data_url, checksum_url)
    parse_fn(csv_bytes) -> DataFrame
    Returns concatenated DataFrame sorted by time.
    """
    months  = list(month_range(start, end))
    dfs     = []
    skipped = []
    failed  = []
    cs_ok   = 0

    log.info(f"\n{'='*60}")
    log.info(f"  {name}")
    log.info(f"  {start.strftime('%Y-%m')} -> {end.strftime('%Y-%m')}  ({len(months)} months)")
    log.info(f"{'='*60}")

    for year, month in tqdm(months, desc=f"  {name}", unit="mo"):
        data_url, cs_url = url_fn(year, month)
        time.sleep(REQUEST_DELAY)

        data = download_bytes(data_url)
        if data is None:
            skipped.append(f"{year}-{month:02d}")
            continue

        cs = download_bytes(cs_url)
        if cs and check_sha256(data, cs):
            cs_ok += 1

        try:
            csv   = unzip_csv(data)
            df    = parse_fn(csv)
            dfs.append(df)
        except Exception as e:
            log.error(f"  Parse error {year}-{month:02d}: {e}")
            failed.append(f"{year}-{month:02d}")

    log.info(f"\n  Result: {len(dfs)} months OK | {len(skipped)} skipped | {len(failed)} failed | {cs_ok} checksums verified")
    if skipped[:5]:
        log.info(f"  Skipped (first 5): {skipped[:5]}")
    if failed:
        log.warning(f"  Failed: {failed}")

    if not dfs:
        return None

    ts_col = "bar_start_ts_ms" if "bar_start_ts_ms" in dfs[0].columns else "fund_calc_time_ms"
    result = pd.concat(dfs, ignore_index=True)
    result = result.sort_values(ts_col).drop_duplicates(subset=[ts_col]).reset_index(drop=True)
    log.info(f"  Total rows: {len(result):,}")
    return result

def month_range(start, end):
    cur = start.replace(day=1)
    while cur <= end:
        yield cur.year, cur.month
        cur += relativedelta(months=1)

# ==============================================================================
# QA
# ==============================================================================

def qa(df, name):
    if df is None or len(df) == 0:
        log.warning(f"  QA WARN: {name} empty")
        return
    log.info(f"\n  QA - {name}:")
    log.info(f"    Rows:     {len(df):,}")

    ts_col = "bar_start_ts_utc" if "bar_start_ts_utc" in df.columns else "fund_time_utc"
    log.info(f"    Range:    {df[ts_col].min()} -> {df[ts_col].max()}")

    if "bar_start_ts_ms" in df.columns:
        dupes = df.duplicated(subset=["bar_start_ts_ms"]).sum()
        diffs = df["bar_start_ts_ms"].diff().dropna()
        gaps  = (diffs > 90_000).sum()
        log.info(f"    Dupes:    {dupes}")
        log.info(f"    Gaps>90s: {gaps}")

    if "meta_ohlc_valid" in df.columns:
        invalid = (~df["meta_ohlc_valid"]).sum()
        log.info(f"    OHLC bad: {invalid}")

    if "fund_rate" in df.columns:
        log.info(f"    Rate min/max: {df['fund_rate'].min():.6f} / {df['fund_rate'].max():.6f}")

# ==============================================================================
# SAVE
# ==============================================================================

def save(df, fname):
    if df is None or len(df) == 0:
        log.warning(f"  SKIP save (empty): {fname}")
        return
    path = RAW_DIR / fname
    df.to_parquet(path, index=False, engine="pyarrow")
    mb = path.stat().st_size / 1024**2
    log.info(f"  Saved: {fname}  ({mb:.1f} MB)")

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    log.info("=" * 60)
    log.info("  BTCDataset v2 - Step 1: Download")
    log.info(f"  Spot:    {SPOT_START.strftime('%Y-%m')} onwards")
    log.info(f"  Perp:    {PERP_START.strftime('%Y-%m')} onwards")
    log.info(f"  Output:  {RAW_DIR}")
    log.info("=" * 60)

    # 1 - Spot 1m
    log.info("\n[1/5] Spot 1m klines")
    spot = download_monthly(
        "Spot 1m",
        lambda y, m: spot_urls(SYMBOL, "1m", y, m),
        lambda csv: parse_kline_csv(csv, "spot", "last"),
        SPOT_START, END_DATE,
    )
    qa(spot, "Spot 1m")
    save(spot, "BTCUSDT_spot_1m_raw.parquet")
    del spot

    # 2 - Perp 1m last price
    log.info("\n[2/5] Perp 1m klines (last price)")
    perp = download_monthly(
        "Perp 1m last",
        lambda y, m: perp_urls(SYMBOL, "1m", y, m, "klines"),
        lambda csv: parse_kline_csv(csv, "perp", "last"),
        PERP_START, END_DATE,
    )
    qa(perp, "Perp 1m last")
    save(perp, "BTCUSDT_perp_1m_raw.parquet")
    del perp

    # 3 - Perp mark price 1m
    log.info("\n[3/5] Perp 1m mark price")
    mark = download_monthly(
        "Perp 1m mark",
        lambda y, m: perp_urls(SYMBOL, "1m", y, m, "markPriceKlines"),
        lambda csv: parse_kline_csv(csv, "perp", "mark"),
        PERP_START, END_DATE,
    )
    qa(mark, "Perp mark 1m")
    save(mark, "BTCUSDT_perp_mark_1m_raw.parquet")
    del mark

    # 4 - Perp index price 1m
    log.info("\n[4/5] Perp 1m index price")
    idx = download_monthly(
        "Perp 1m index",
        lambda y, m: perp_urls(SYMBOL, "1m", y, m, "indexPriceKlines"),
        lambda csv: parse_kline_csv(csv, "perp", "index"),
        PERP_START, END_DATE,
    )
    qa(idx, "Perp index 1m")
    save(idx, "BTCUSDT_perp_index_1m_raw.parquet")
    del idx

    # 5 - Funding rate
    log.info("\n[5/5] Funding rate history")
    fund = download_monthly(
        "Funding rate",
        lambda y, m: funding_urls(SYMBOL, y, m),
        parse_funding_csv,
        PERP_START, END_DATE,
    )
    qa(fund, "Funding rate")
    save(fund, "BTCUSDT_perp_funding_raw.parquet")
    del fund

    # Final summary
    log.info("\n" + "=" * 60)
    log.info("  Step 1 complete.")
    log.info(f"  Files in {RAW_DIR}:")
    for f in sorted(RAW_DIR.glob("*.parquet")):
        mb = f.stat().st_size / 1024**2
        log.info(f"    {f.name:<50} {mb:>8.1f} MB")
    log.info("\n  Next: run scripts/resample_v2.py")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
