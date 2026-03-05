"""
download_aggtrades.py -- Download Binance aggTrades and compute true tick CVD
D37b: True Tick CVD Pipeline for BTCDataset_v2

Downloads monthly aggTrades from data.binance.vision, streams CSV in chunks
to compute per-5m-bar cumulative volume delta (CVD) from actual tick data.
This replaces the CLV-based CVD proxy with true buyer/seller volume.

URL pattern:
  https://data.binance.vision/data/futures/um/monthly/aggTrades/BTCUSDT/
    BTCUSDT-aggTrades-YYYY-MM.zip
  Checksum: same URL + .CHECKSUM

Output: core/data/raw/aggtrades/aggtrades_cvd_YYYY-MM.parquet (one per month)

WARNING: Monthly files are 100 MB - 1.3 GB compressed. Full run takes 4-8 hours.
         Files are streamed to disk and processed in chunks to avoid RAM issues.

Usage: python data_pipeline/download_aggtrades.py
       python data_pipeline/download_aggtrades.py --test  (one month only)
"""

import os
import sys
import time
import hashlib
import zipfile
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "core", "data", "raw", "aggtrades")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
BASE_URL = (
    "https://data.binance.vision/data/futures/um/monthly/aggTrades/BTCUSDT"
)

START_YEAR, START_MONTH = 2020, 1
END_YEAR, END_MONTH = 2026, 2

CHUNK_SIZE = 100_000          # rows per CSV chunk
ZSCORE_WINDOW = 20            # rolling window for z-score
BAR_MS = 5 * 60 * 1000       # 5 minutes in milliseconds
DOWNLOAD_CHUNK = 1024 * 1024  # 1 MB chunks for streaming download
CONNECT_TIMEOUT = 15          # seconds
READ_TIMEOUT = 120            # seconds per chunk (large files)
MAX_RETRIES = 1               # retries per month on transient errors

# Session boundary hours (UTC) for session-reset CVD
# Asia: 00:00-07:55, London: 08:00-12:55, NY: 13:00-23:55
SESSION_HOURS = [0, 8, 13]

# aggTrades CSV columns (Binance format)
AGG_COLS = [
    "agg_trade_id", "price", "quantity",
    "first_trade_id", "last_trade_id",
    "transact_time", "is_buyer_maker",
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "download_aggtrades.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def month_range(start_y, start_m, end_y, end_m):
    """Generate (year, month) tuples from start to end inclusive."""
    y, m = start_y, start_m
    while (y, m) <= (end_y, end_m):
        yield y, m
        m += 1
        if m > 12:
            m = 1
            y += 1


def output_path(year, month):
    """Path to output parquet for a given month."""
    return os.path.join(
        OUTPUT_DIR, f"aggtrades_cvd_{year:04d}-{month:02d}.parquet"
    )


def ym_str(year, month):
    """Format year-month as string."""
    return f"{year:04d}-{month:02d}"


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------
def download_checksum(session, year, month):
    """Download SHA256 checksum. Returns (hash_str, None) or (None, error)."""
    fn = f"BTCUSDT-aggTrades-{ym_str(year, month)}.zip"
    url = f"{BASE_URL}/{fn}.CHECKSUM"
    try:
        r = session.get(url, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
        if r.status_code == 404:
            return None, "checksum not found (404)"
        if r.status_code != 200:
            return None, f"checksum HTTP {r.status_code}"
        return r.text.strip().split()[0], None
    except requests.RequestException as e:
        return None, f"checksum error: {e}"


def download_zip_streaming(session, year, month, dest_path):
    """Stream-download ZIP to disk file. Returns (file_size, None) or (None, error)."""
    fn = f"BTCUSDT-aggTrades-{ym_str(year, month)}.zip"
    url = f"{BASE_URL}/{fn}"
    try:
        r = session.get(
            url, stream=True, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)
        )
        if r.status_code == 404:
            return None, "zip not found (404)"
        if r.status_code != 200:
            return None, f"zip HTTP {r.status_code}"

        total = 0
        with open(dest_path, "wb") as f:
            for data in r.iter_content(chunk_size=DOWNLOAD_CHUNK):
                f.write(data)
                total += len(data)
        return total, None
    except requests.RequestException as e:
        return None, f"download error: {e}"


def verify_sha256(file_path, expected_hash):
    """Verify SHA256 of a file on disk. Returns True if match."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            data = f.read(65536)
            if not data:
                break
            h.update(data)
    actual = h.hexdigest()
    return actual == expected_hash


def detect_has_header(zip_path):
    """Check if the CSV inside the ZIP has a header row."""
    with zipfile.ZipFile(zip_path) as zf:
        csv_name = zf.namelist()[0]
        with zf.open(csv_name) as f:
            first_line = f.readline().decode("utf-8").strip()
    # If first field is numeric, there is no header
    first_field = first_line.split(",")[0]
    try:
        int(first_field)
        return False
    except ValueError:
        return True


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------
def process_month_zip(zip_path):
    """Stream-process a monthly aggTrades ZIP into per-5m-bar CVD.

    Returns (DataFrame, rows_processed) or (None, rows_processed).
    DataFrame columns: bar_start_ts_utc, cvd_true_bar
    """
    has_header = detect_has_header(zip_path)

    # Collect per-bar delta sums from each chunk
    bar_sum_parts = []
    rows_processed = 0

    with zipfile.ZipFile(zip_path) as zf:
        csv_name = zf.namelist()[0]
        with zf.open(csv_name) as f:
            reader_kwargs = {
                "chunksize": CHUNK_SIZE,
                "usecols": [2, 5, 6],  # quantity, transact_time, is_buyer_maker
            }

            if has_header:
                reader_kwargs["header"] = 0
            else:
                reader_kwargs["header"] = None
                reader_kwargs["names"] = AGG_COLS

            for chunk in pd.read_csv(f, **reader_kwargs):
                qty = chunk.iloc[:, 0] if not has_header else chunk["quantity"]
                ts = chunk.iloc[:, 1] if not has_header else chunk["transact_time"]
                ibm = chunk.iloc[:, 2] if not has_header else chunk["is_buyer_maker"]

                # Normalize is_buyer_maker to boolean
                if ibm.dtype == object:
                    ibm = ibm.astype(str).str.lower().isin(["true", "1"])
                ibm = ibm.astype(bool)

                # delta: positive = buyer-initiated, negative = seller-initiated
                qty_vals = pd.to_numeric(qty, errors="coerce").fillna(0.0).values
                delta = qty_vals * np.where(ibm.values, -1.0, 1.0)

                # Floor timestamp to 5m bar
                ts_vals = pd.to_numeric(ts, errors="coerce").fillna(0).astype(np.int64)
                bar_ts = (ts_vals // BAR_MS) * BAR_MS

                # Group by bar and sum
                chunk_df = pd.DataFrame({
                    "bar_start_ts_ms": bar_ts,
                    "delta": delta,
                })
                chunk_sum = (
                    chunk_df.groupby("bar_start_ts_ms")["delta"]
                    .sum()
                    .reset_index()
                )
                bar_sum_parts.append(chunk_sum)
                rows_processed += len(chunk)

    if not bar_sum_parts:
        return None, rows_processed

    # Aggregate all chunk sums by bar
    all_sums = pd.concat(bar_sum_parts, ignore_index=True)
    del bar_sum_parts

    result = (
        all_sums.groupby("bar_start_ts_ms")["delta"]
        .sum()
        .reset_index()
    )
    result.columns = ["bar_start_ts_ms", "cvd_true_bar"]
    result = result.sort_values("bar_start_ts_ms").reset_index(drop=True)

    # Convert timestamp
    result["bar_start_ts_utc"] = pd.to_datetime(
        result["bar_start_ts_ms"], unit="ms", utc=True
    )
    result = result.drop(columns=["bar_start_ts_ms"])

    return result, rows_processed


def compute_derived(df):
    """Compute daily-reset, session-reset, and z-score CVD columns.

    Input must have: bar_start_ts_utc, cvd_true_bar
    Output adds: cvd_true_daily, cvd_true_session, cvd_true_zscore
    """
    # cvd_true_daily: cumsum within each calendar day (reset at 00:00 UTC)
    day_key = df["bar_start_ts_utc"].dt.floor("D")
    df["cvd_true_daily"] = df.groupby(day_key)["cvd_true_bar"].cumsum()

    # cvd_true_session: cumsum within each session
    # Sessions: Asia [0,8), London [8,13), NY [13,24)
    hour = df["bar_start_ts_utc"].dt.hour
    session_id = np.select(
        [hour < 8, hour < 13],
        [0, 1],
        default=2,
    )
    session_key = day_key.astype(str).values + "_" + session_id.astype(str)
    df["cvd_true_session"] = df.groupby(session_key)["cvd_true_bar"].cumsum()

    # cvd_true_zscore: rolling z-score of cvd_true_daily
    rm = df["cvd_true_daily"].rolling(ZSCORE_WINDOW).mean()
    rs = df["cvd_true_daily"].rolling(ZSCORE_WINDOW).std()
    df["cvd_true_zscore"] = (df["cvd_true_daily"] - rm) / rs

    # Final column order
    df = df[[
        "bar_start_ts_utc", "cvd_true_bar", "cvd_true_daily",
        "cvd_true_session", "cvd_true_zscore",
    ]]

    return df


# ---------------------------------------------------------------------------
# Correlation validation
# ---------------------------------------------------------------------------
def validate_correlation(all_cvd):
    """Compare true tick CVD with existing CLV-based CVD from v2 parquet."""
    v2_path = os.path.join(
        PROJECT_ROOT, "data", "labeled", "BTCUSDT_5m_labeled_v2.parquet"
    )
    if not os.path.exists(v2_path):
        print("  v2 parquet not found -- skipping correlation check")
        return

    print("  Loading v2 columns for comparison...")
    v2 = pd.read_parquet(
        v2_path, columns=["bar_start_ts_utc", "cvd_zscore", "cvd_bar"]
    )
    merged = v2.merge(all_cvd, on="bar_start_ts_utc", how="inner")

    if len(merged) < 1000:
        print(f"  Only {len(merged)} overlapping bars -- too few for correlation")
        return

    # Bar-level correlation
    mask_b = merged["cvd_bar"].notna() & merged["cvd_true_bar"].notna()
    corr_bar = merged.loc[mask_b, "cvd_bar"].corr(
        merged.loc[mask_b, "cvd_true_bar"]
    )
    print(f"  Pearson(cvd_bar, cvd_true_bar)       = {corr_bar:.4f}")

    # Z-score correlation
    mask_z = merged["cvd_zscore"].notna() & merged["cvd_true_zscore"].notna()
    if mask_z.sum() > 100:
        corr_z = merged.loc[mask_z, "cvd_zscore"].corr(
            merged.loc[mask_z, "cvd_true_zscore"]
        )
        print(f"  Pearson(cvd_zscore, cvd_true_zscore) = {corr_z:.4f}")

    print()
    if corr_bar < 0.5:
        print("  WARNING: cvd_true_bar correlation below 0.5 -- possible bug!")
        print("  Expected range: 0.5-0.7. Stop and investigate.")
    elif corr_bar > 0.7:
        print("  NOTE: correlation above 0.7 -- tick CVD closely tracks CLV proxy")
    else:
        print("  OK: bar-level correlation in expected range (0.5-0.7)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    test_mode = "--test" in sys.argv

    print("=" * 60)
    print("download_aggtrades.py -- True Tick CVD Pipeline (D37b)")
    print("=" * 60)
    print()

    if test_mode:
        print("*** TEST MODE: processing 2020-01 only ***")
        print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Build month list
    if test_mode:
        months = [(2020, 1)]
    else:
        months = list(month_range(START_YEAR, START_MONTH, END_YEAR, END_MONTH))

    print(f"Date range : {ym_str(*months[0])} to {ym_str(*months[-1])}")
    print(f"Months     : {len(months)}")
    print(f"Output dir : {OUTPUT_DIR}")
    print()

    # Resume support: skip months that already have output
    already_done = []
    to_process = []
    for y, m in months:
        if os.path.exists(output_path(y, m)):
            already_done.append((y, m))
        else:
            to_process.append((y, m))

    if already_done:
        print(f"Already done: {len(already_done)} months (skipping)")
    print(f"To process : {len(to_process)} months")
    print()

    if not to_process:
        print("All months already downloaded. Nothing to do.")
        # Still run validation if we have data
        _run_final_validation(months)
        return

    # -------------------------------------------------------------------
    # Process each month
    # -------------------------------------------------------------------
    http = requests.Session()
    total_bars = 0
    total_trades = 0
    failed = []
    t0 = time.time()

    for idx, (year, month) in enumerate(to_process):
        ym = ym_str(year, month)
        print(f"[{idx+1}/{len(to_process)}] {ym}")

        success = False
        for attempt in range(MAX_RETRIES + 1):
            if attempt > 0:
                print(f"  Retry {attempt}/{MAX_RETRIES}...")
                time.sleep(5)

            result = _process_one_month(http, year, month)
            if result is not None:
                bars, trades = result
                total_bars += bars
                total_trades += trades
                success = True
                break
            # On 404 or parse error, do not retry
            break

        if not success:
            failed.append(ym)

        print()

    http.close()
    elapsed = time.time() - t0

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    print("=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print()
    print(f"Months processed : {len(to_process) - len(failed)}/{len(to_process)}")
    print(f"Failed months    : {len(failed)}")
    print(f"Total 5m bars    : {total_bars:,}")
    print(f"Total trades     : {total_trades:,}")
    print(f"Elapsed          : {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    print()

    if failed:
        print("Failed months:")
        for ym_f in failed:
            print(f"  {ym_f}")
        print()

    # Final validation
    _run_final_validation(months)

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)


def _process_one_month(http, year, month):
    """Download and process one month. Returns (n_bars, n_trades) or None."""
    ym = ym_str(year, month)
    tmp_path = os.path.join(OUTPUT_DIR, f"_tmp_{ym}.zip")

    try:
        # 1. Download checksum
        expected_hash, err = download_checksum(http, year, month)
        if expected_hash is None:
            print(f"  SKIP: {err}")
            log.warning("Skip %s: %s", ym, err)
            return None

        # 2. Stream-download ZIP to disk
        sys.stdout.write("  Downloading... ")
        sys.stdout.flush()
        t1 = time.time()
        fsize, err = download_zip_streaming(http, year, month, tmp_path)
        if fsize is None:
            print(f"FAIL: {err}")
            log.warning("Download fail %s: %s", ym, err)
            return None
        dl_sec = time.time() - t1
        mb = fsize / (1024 * 1024)
        print(f"{mb:.0f} MB in {dl_sec:.0f}s ({mb/dl_sec:.1f} MB/s)")

        # 3. Verify SHA256
        sys.stdout.write("  Verifying SHA256... ")
        sys.stdout.flush()
        if not verify_sha256(tmp_path, expected_hash):
            print("MISMATCH -- skipping")
            log.error("SHA256 mismatch for %s", ym)
            return None
        print("OK")

        # 4. Process trades into bars
        sys.stdout.write("  Computing CVD... ")
        sys.stdout.flush()
        t2 = time.time()
        df, n_trades = process_month_zip(tmp_path)
        proc_sec = time.time() - t2

        if df is None or len(df) == 0:
            print(f"no data ({n_trades:,} raw rows)")
            log.warning("No bars for %s (%d trades)", ym, n_trades)
            return None

        # 5. Compute derived features
        df = compute_derived(df)
        n_bars = len(df)
        print(f"{n_trades:,} trades -> {n_bars:,} bars in {proc_sec:.0f}s")

        # 6. Save parquet
        out = output_path(year, month)
        df.to_parquet(out, engine="pyarrow", compression="zstd", index=False)
        fsize_out = os.path.getsize(out) / 1024
        print(f"  Saved: {out} ({fsize_out:.0f} KB)")

        log.info(
            "Saved %s: %d bars from %d trades in %.0fs",
            ym, n_bars, n_trades, proc_sec,
        )
        return n_bars, n_trades

    finally:
        # Always clean up temp file
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _run_final_validation(months):
    """Load all output parquets and run validation."""
    print("-" * 60)
    print("VALIDATION")
    print("-" * 60)
    print()

    # Load all output parquets
    parts = []
    for y, m in months:
        p = output_path(y, m)
        if os.path.exists(p):
            parts.append(pd.read_parquet(p))

    if not parts:
        print("No output files found -- nothing to validate")
        return

    all_cvd = pd.concat(parts, ignore_index=True)
    del parts
    all_cvd = all_cvd.sort_values("bar_start_ts_utc").reset_index(drop=True)

    print(f"Total bars across all months: {len(all_cvd):,}")
    ts_min = all_cvd["bar_start_ts_utc"].min()
    ts_max = all_cvd["bar_start_ts_utc"].max()
    print(f"Date range: {ts_min} to {ts_max}")
    print(f"Columns: {list(all_cvd.columns)}")
    print()

    # NaN rates
    print("NaN rates:")
    for col in all_cvd.columns:
        if col == "bar_start_ts_utc":
            continue
        nan_pct = all_cvd[col].isna().mean() * 100
        print(f"  {col:25s}: {nan_pct:.2f}%%")
    print()

    # Midnight rollover sample (confirm daily reset)
    print("5 rows at midnight UTC rollover (confirm daily reset):")
    midnight_mask = (
        (all_cvd["bar_start_ts_utc"].dt.hour == 0)
        & (all_cvd["bar_start_ts_utc"].dt.minute == 0)
    )
    midnight_idxs = all_cvd.index[midnight_mask]

    shown = 0
    for mi in midnight_idxs:
        if mi > 0 and shown < 5:
            row_before = all_cvd.iloc[mi - 1]
            row_at = all_cvd.iloc[mi]
            print(
                f"  {row_before['bar_start_ts_utc']}  "
                f"daily={row_before['cvd_true_daily']:>12.1f}"
            )
            print(
                f"  {row_at['bar_start_ts_utc']}  "
                f"daily={row_at['cvd_true_daily']:>12.1f}  <-- reset"
            )
            print()
            shown += 1
    if shown == 0:
        print("  (no midnight bars found)")
    print()

    # Pearson correlation with existing CLV-based CVD
    print("Pearson correlation vs existing cvd (CLV proxy):")
    try:
        validate_correlation(all_cvd)
    except Exception as e:
        print(f"  Correlation check failed: {e}")

    # Rows per month
    print()
    print("Rows per month (first/last 3):")
    all_cvd["_ym"] = (
        all_cvd["bar_start_ts_utc"].dt.tz_localize(None).dt.to_period("M")
    )
    counts = all_cvd.groupby("_ym").size()
    for ym_val in list(counts.index[:3]) + ["..."] + list(counts.index[-3:]):
        if ym_val == "...":
            print("  ...")
        else:
            print(f"  {ym_val}: {counts[ym_val]:,} bars")


if __name__ == "__main__":
    main()
