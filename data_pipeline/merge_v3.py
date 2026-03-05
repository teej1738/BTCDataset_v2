"""
merge_v3.py -- Merge v2 parquet with OI, CVD, and liquidation data to create v3
D37d: Dataset Merge v2 -> v3 for BTCDataset_v2

Reads v2 as source (READ ONLY). Merges new columns from three data sources.
Writes new v3 parquet. Never overwrites v2.

INPUT:
  Source: data/labeled/BTCUSDT_5m_labeled_v2.parquet (READ ONLY)
  OI:    core/data/raw/oi_metrics/*.parquet  (5m bars, 2021-12 to 2026-03)
  CVD:   core/data/raw/aggtrades/*.parquet   (5m bars, 2020-01 to 2026-02)
  Liq:   core/data/raw/liquidations/*.parquet (DAILY, 2020-01 to 2026-03)

OUTPUT:
  data/labeled/BTCUSDT_5m_labeled_v3.parquet
  data/labeled/feature_catalog_v3.yaml

Merge strategy:
  - Left join on bar_start_ts_utc, keeping all v2 rows
  - NaN where source data is missing
  - OI timestamps cast from datetime64[us] to datetime64[ms] for alignment
  - Liquidation daily data: shifted +1 day for causality, forward-filled to 5m
    (day D's liquidations available from day D+1 00:00 UTC)
  - Liquidation BTC values converted to USD using bar-level close price

New columns (25 total):
  OI  (10): oi_btc, oi_usdt, toptrader_ls_ratio_count,
            toptrader_ls_ratio_position, global_ls_ratio, taker_ls_vol_ratio,
            oi_change_1h, oi_change_4h, oi_change_pct_1h, oi_zscore_20
  CVD  (4): cvd_true_bar, cvd_true_daily, cvd_true_session, cvd_true_zscore
  Liq  (8): liq_long_btc, liq_short_btc, liq_total_btc, liq_ratio,
            liq_cascade_flag, liq_zscore_7d, liq_change_1d, liq_change_pct_1d
  USD  (3): liq_long_usd, liq_short_usd, liq_total_usd

Usage: python data_pipeline/merge_v3.py
"""

import os
import sys
import time
import glob
import logging

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V2_PATH = os.path.join(
    PROJECT_ROOT, "data", "labeled", "BTCUSDT_5m_labeled_v2.parquet"
)
V3_PATH = os.path.join(
    PROJECT_ROOT, "data", "labeled", "BTCUSDT_5m_labeled_v3.parquet"
)
CATALOG_V2 = os.path.join(
    PROJECT_ROOT, "data", "labeled", "feature_catalog_v2.yaml"
)
CATALOG_V3 = os.path.join(
    PROJECT_ROOT, "data", "labeled", "feature_catalog_v3.yaml"
)

OI_DIR = os.path.join(PROJECT_ROOT, "core", "data", "raw", "oi_metrics")
CVD_DIR = os.path.join(PROJECT_ROOT, "core", "data", "raw", "aggtrades")
LIQ_DIR = os.path.join(PROJECT_ROOT, "core", "data", "raw", "liquidations")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

EXPECTED_V2_ROWS = 648_288

# Causality gate test points
CAUSALITY_TEST_POINTS = [1000, 5000, 50000]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "merge_v3.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_monthly_parquets(directory, name):
    """Load and concatenate all monthly parquet files from a directory."""
    pattern = os.path.join(directory, "*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"ERROR: No parquet files in {directory}")
        sys.exit(1)

    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values("bar_start_ts_utc").reset_index(drop=True)

    n_before = len(df)
    df = df.drop_duplicates(subset=["bar_start_ts_utc"], keep="first")
    n_dupes = n_before - len(df)

    ts_min = df["bar_start_ts_utc"].min()
    ts_max = df["bar_start_ts_utc"].max()
    print(
        f"  {name:4s}: {len(files):3d} files, {len(df):>9,} rows  "
        f"({str(ts_min)[:10]} to {str(ts_max)[:10]})"
    )
    if n_dupes:
        print(f"        Removed {n_dupes} duplicate timestamps")
    return df


def write_feature_catalog_v3(
    v2_catalog_path, v3_catalog_path,
    oi_cols, cvd_cols, liq_btc_cols, liq_usd_cols,
):
    """Create v3 feature catalog by extending v2 catalog."""
    v2_content = ""
    if os.path.exists(v2_catalog_path):
        with open(v2_catalog_path, encoding="utf-8") as f:
            v2_content = f.read()

    lines = []
    lines.append("")
    lines.append("# ============================================================")
    lines.append("# v3 new features (D37d merge)")
    lines.append("# Added by merge_v3.py")
    lines.append("# Total new columns: 25")
    lines.append("# ============================================================")
    lines.append("")

    # OI columns
    oi_desc = {
        "oi_btc": ("OI/Raw", "Open interest in BTC", 0),
        "oi_usdt": ("OI/Raw", "Open interest in USDT", 0),
        "toptrader_ls_ratio_count": (
            "OI/Ratios", "Top trader long/short ratio by count", 0
        ),
        "toptrader_ls_ratio_position": (
            "OI/Ratios", "Top trader long/short ratio by position", 0
        ),
        "global_ls_ratio": ("OI/Ratios", "Global long/short ratio", 0),
        "taker_ls_vol_ratio": (
            "OI/Ratios", "Taker long/short volume ratio", 0
        ),
        "oi_change_1h": ("OI/Derived", "OI absolute change over 1h (12 bars)", 12),
        "oi_change_4h": ("OI/Derived", "OI absolute change over 4h (48 bars)", 48),
        "oi_change_pct_1h": (
            "OI/Derived", "OI percentage change over 1h (%)", 12
        ),
        "oi_zscore_20": ("OI/Derived", "OI z-score (20-bar rolling window)", 20),
    }
    lines.append("# --- OI metrics (Binance, 5m bars, from 2021-12-01) ---")
    lines.append("# Source: data_pipeline/download_oi.py (D37a)")
    lines.append("# NaN for all bars before 2021-12-01")
    lines.append("")
    for col in oi_cols:
        fam, desc, warmup = oi_desc.get(col, ("OI", col, 0))
        lines.append(f"{col}:")
        lines.append(f"  family: {fam}")
        lines.append(f"  description: {desc}")
        lines.append(f"  warmup_bars: {warmup}")
        lines.append(f"  evidence_tier: 2")
        lines.append(f"  computation_date: 2026-03-03")
        lines.append("")

    # CVD columns
    cvd_desc = {
        "cvd_true_bar": (
            "CVD/True", "True tick CVD per 5m bar (buyer - seller BTC)", 0
        ),
        "cvd_true_daily": (
            "CVD/True", "True tick CVD cumulative, resets 00:00 UTC daily", 0
        ),
        "cvd_true_session": (
            "CVD/True", "True tick CVD cumulative, resets at session opens", 0
        ),
        "cvd_true_zscore": (
            "CVD/True", "True tick CVD daily z-score (20-bar rolling)", 20
        ),
    }
    lines.append("# --- True tick CVD (Binance aggTrades, 5m, 2020-01 to 2026-02) ---")
    lines.append("# Source: data_pipeline/download_aggtrades.py (D37b)")
    lines.append("# Computed from actual trade-level buyer/seller volume")
    lines.append("")
    for col in cvd_cols:
        fam, desc, warmup = cvd_desc.get(col, ("CVD", col, 0))
        lines.append(f"{col}:")
        lines.append(f"  family: {fam}")
        lines.append(f"  description: {desc}")
        lines.append(f"  warmup_bars: {warmup}")
        lines.append(f"  evidence_tier: 2")
        lines.append(f"  computation_date: 2026-03-03")
        lines.append("")

    # Liq BTC columns
    liq_desc = {
        "liq_long_btc": (
            "Liq/Raw", "Daily long liquidation volume (BTC), 1-day lagged", 0
        ),
        "liq_short_btc": (
            "Liq/Raw", "Daily short liquidation volume (BTC), 1-day lagged", 0
        ),
        "liq_total_btc": (
            "Liq/Raw", "Daily total liquidation volume (BTC), 1-day lagged", 0
        ),
        "liq_ratio": (
            "Liq/Derived", "Fraction of liq that are longs, 1-day lagged", 0
        ),
        "liq_cascade_flag": (
            "Liq/Derived",
            "1 when liq_total > 3x 7-day rolling mean, 1-day lagged", 0
        ),
        "liq_zscore_7d": (
            "Liq/Derived",
            "Liq total z-score (7-day rolling), 1-day lagged", 0
        ),
        "liq_change_1d": (
            "Liq/Derived",
            "1-day change in daily liq total (BTC), 1-day lagged", 0
        ),
        "liq_change_pct_1d": (
            "Liq/Derived",
            "1-day pct change in daily liq total, 1-day lagged", 0
        ),
    }
    lines.append("# --- Liquidation data (Coinalyze, DAILY, 2020-01 to 2026-03) ---")
    lines.append("# Source: data_pipeline/download_liquidations.py (D37c)")
    lines.append("# Daily -> 5m forward-fill with +1 day causality shift")
    lines.append("# BTC values from API, USD = BTC * close at each bar")
    lines.append("")
    for col in liq_btc_cols:
        fam, desc, warmup = liq_desc.get(col, ("Liq", col, 0))
        lines.append(f"{col}:")
        lines.append(f"  family: {fam}")
        lines.append(f"  description: {desc}")
        lines.append(f"  warmup_bars: {warmup}")
        lines.append(f"  granularity: daily_ffill")
        lines.append(f"  evidence_tier: 2")
        lines.append(f"  computation_date: 2026-03-03")
        lines.append("")

    # Liq USD columns
    liq_usd_desc = {
        "liq_long_usd": (
            "Liq/USD",
            "Daily long liq USD (= liq_long_btc * close), 1-day lagged", 0
        ),
        "liq_short_usd": (
            "Liq/USD",
            "Daily short liq USD (= liq_short_btc * close), 1-day lagged", 0
        ),
        "liq_total_usd": (
            "Liq/USD",
            "Daily total liq USD (= liq_total_btc * close), 1-day lagged", 0
        ),
    }
    for col in liq_usd_cols:
        fam, desc, warmup = liq_usd_desc.get(col, ("Liq/USD", col, 0))
        lines.append(f"{col}:")
        lines.append(f"  family: {fam}")
        lines.append(f"  description: {desc}")
        lines.append(f"  warmup_bars: {warmup}")
        lines.append(f"  granularity: daily_ffill")
        lines.append(f"  evidence_tier: 2")
        lines.append(f"  computation_date: 2026-03-03")
        lines.append("")

    with open(v3_catalog_path, "w", encoding="utf-8") as f:
        if v2_content:
            f.write(v2_content)
            if not v2_content.endswith("\n"):
                f.write("\n")
        f.write("\n".join(lines))

    total_lines = (v2_content + "\n".join(lines)).count("\n") + 1
    print(f"  Feature catalog v3 written: {v3_catalog_path}")
    print(f"  Total lines: {total_lines}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("merge_v3.py -- Dataset Merge v2 -> v3")
    print("=" * 60)
    print()

    t0 = time.time()

    # -------------------------------------------------------------------
    # Phase 0: Pre-checks
    # -------------------------------------------------------------------
    print("Phase 0: Pre-checks...")
    if not os.path.exists(V2_PATH):
        print(f"ERROR: v2 not found at {V2_PATH}")
        sys.exit(1)

    v2_mtime = os.path.getmtime(V2_PATH)
    v2_size = os.path.getsize(V2_PATH)
    print(f"  v2 path : {V2_PATH}")
    print(f"  v2 size : {v2_size / 1e6:.1f} MB")

    for d, name in [(OI_DIR, "OI"), (CVD_DIR, "CVD"), (LIQ_DIR, "Liq")]:
        if not os.path.exists(d):
            print(f"ERROR: {name} dir not found: {d}")
            sys.exit(1)

    if os.path.exists(V3_PATH):
        print(f"  NOTE: v3 already exists -- will be overwritten")
    print()

    # -------------------------------------------------------------------
    # Phase 1: Load all sources
    # -------------------------------------------------------------------
    print("Phase 1: Loading sources...")
    print()
    print("  Loading v2 parquet...")
    v2 = pd.read_parquet(V2_PATH)
    v2_rows = len(v2)
    v2_cols = len(v2.columns)
    assert v2_rows == EXPECTED_V2_ROWS, (
        f"v2 row count mismatch: {v2_rows} vs expected {EXPECTED_V2_ROWS}"
    )
    print(
        f"  v2  : {v2_rows:>9,} rows x {v2_cols} cols  "
        f"({str(v2['bar_start_ts_utc'].min())[:10]} to "
        f"{str(v2['bar_start_ts_utc'].max())[:10]})"
    )
    print()

    oi = load_monthly_parquets(OI_DIR, "OI")
    cvd = load_monthly_parquets(CVD_DIR, "CVD")
    liq = load_monthly_parquets(LIQ_DIR, "Liq")
    print()

    # -------------------------------------------------------------------
    # Phase 2: Timestamp alignment
    # -------------------------------------------------------------------
    print("Phase 2: Timestamp alignment...")

    v2_ts_dtype = v2["bar_start_ts_utc"].dtype
    oi_ts_dtype = oi["bar_start_ts_utc"].dtype
    print(f"  v2  ts dtype: {v2_ts_dtype}")
    print(f"  OI  ts dtype: {oi_ts_dtype}")
    print(f"  CVD ts dtype: {cvd['bar_start_ts_utc'].dtype}")
    print(f"  Liq ts dtype: {liq['bar_start_ts_utc'].dtype}")

    # Cast OI from datetime64[us, UTC] to datetime64[ms, UTC] if needed
    if str(oi_ts_dtype) != str(v2_ts_dtype):
        oi["bar_start_ts_utc"] = oi["bar_start_ts_utc"].dt.as_unit("ms")
        print(f"  OI  ts cast -> {oi['bar_start_ts_utc'].dtype}")
    print()

    # -------------------------------------------------------------------
    # Phase 3: Merge OI (left join on bar_start_ts_utc)
    # -------------------------------------------------------------------
    print("Phase 3: Merging OI data (left join on bar_start_ts_utc)...")

    oi_cols = [c for c in oi.columns if c != "bar_start_ts_utc"]
    print(f"  New OI columns ({len(oi_cols)}): {oi_cols}")

    v3 = v2.merge(
        oi[["bar_start_ts_utc"] + oi_cols],
        on="bar_start_ts_utc",
        how="left",
    )

    oi_matched = v3[oi_cols[0]].notna().sum()
    pct = oi_matched / v2_rows * 100
    print(f"  OI matched: {oi_matched:,} / {v2_rows:,} bars ({pct:.1f}%)")
    assert len(v3) == v2_rows, (
        f"Row count changed after OI merge: {len(v3)} vs {v2_rows}"
    )
    print(f"  Row count preserved: {len(v3):,}")
    print()
    del oi

    # -------------------------------------------------------------------
    # Phase 4: Merge CVD (left join on bar_start_ts_utc)
    # -------------------------------------------------------------------
    print("Phase 4: Merging CVD data (left join on bar_start_ts_utc)...")

    cvd_cols = [c for c in cvd.columns if c != "bar_start_ts_utc"]
    print(f"  New CVD columns ({len(cvd_cols)}): {cvd_cols}")

    v3 = v3.merge(
        cvd[["bar_start_ts_utc"] + cvd_cols],
        on="bar_start_ts_utc",
        how="left",
    )

    cvd_matched = v3[cvd_cols[0]].notna().sum()
    pct = cvd_matched / v2_rows * 100
    print(f"  CVD matched: {cvd_matched:,} / {v2_rows:,} bars ({pct:.1f}%)")
    assert len(v3) == v2_rows, (
        f"Row count changed after CVD merge: {len(v3)} vs {v2_rows}"
    )
    print(f"  Row count preserved: {len(v3):,}")
    print()
    del cvd

    # -------------------------------------------------------------------
    # Phase 5: Merge Liquidations (daily -> 5m, +1 day causality lag)
    # -------------------------------------------------------------------
    print("Phase 5: Merging liquidation data...")
    print("  Strategy: daily liq data shifted +1 day, joined by date")
    print("  (day D's liquidations available from day D+1 00:00 UTC)")

    liq_btc_cols = [c for c in liq.columns if c != "bar_start_ts_utc"]
    print(f"  Liq BTC columns ({len(liq_btc_cols)}): {liq_btc_cols}")

    # Keep original liq event date for causality verification
    liq["_liq_event_date"] = liq["bar_start_ts_utc"].dt.normalize()

    # Shift +1 day: data from day D available starting day D+1
    liq["_merge_date"] = liq["_liq_event_date"] + pd.Timedelta(days=1)

    # Prepare for merge
    liq_for_merge = liq[
        ["_merge_date", "_liq_event_date"] + liq_btc_cols
    ].copy()

    # Verify no duplicate merge dates
    assert liq_for_merge["_merge_date"].is_unique, (
        "Duplicate liq merge dates found!"
    )

    # Create merge key in v3
    v3["_merge_date"] = v3["bar_start_ts_utc"].dt.normalize()

    # Left merge on date (many v3 bars to one liq row per day)
    v3 = v3.merge(liq_for_merge, on="_merge_date", how="left")

    assert len(v3) == v2_rows, (
        f"Row count changed after Liq merge: {len(v3)} vs {v2_rows}"
    )

    liq_matched = v3["liq_total_btc"].notna().sum()
    pct = liq_matched / v2_rows * 100
    print(f"  Liq matched: {liq_matched:,} / {v2_rows:,} bars ({pct:.1f}%)")

    # USD conversion: BTC values * bar-level close price
    print("  Converting BTC -> USD using bar-level close price...")
    liq_usd_cols = ["liq_long_usd", "liq_short_usd", "liq_total_usd"]
    v3["liq_long_usd"] = v3["liq_long_btc"] * v3["close"]
    v3["liq_short_usd"] = v3["liq_short_btc"] * v3["close"]
    v3["liq_total_usd"] = v3["liq_total_btc"] * v3["close"]
    print(f"  Added: {liq_usd_cols}")
    print()

    del liq, liq_for_merge

    # -------------------------------------------------------------------
    # Phase 6: Causality Gate
    # -------------------------------------------------------------------
    print("Phase 6: Causality gate...")
    print(f"  Test points: T = {CAUSALITY_TEST_POINTS}")

    all_new_cols = oi_cols + cvd_cols + liq_btc_cols + liq_usd_cols
    print(f"  Columns to verify: {len(all_new_cols)}")
    causality_pass = True

    # Test 1: Liq date-shift causality
    # Every bar with liq data must have liq_event_date < bar_date
    print("  Test 1: Liq date-shift verification...")
    liq_present = v3["_liq_event_date"].notna()
    if liq_present.any():
        bar_dates = v3.loc[liq_present, "_merge_date"]
        event_dates = v3.loc[liq_present, "_liq_event_date"]
        violations = (event_dates >= bar_dates).sum()
        if violations > 0:
            print(
                f"    FAIL: {violations} bars have liq data from "
                f"same/future day"
            )
            causality_pass = False
        else:
            print(
                f"    PASS: all {liq_present.sum():,} bars use "
                f"previous-day liq data"
            )
    else:
        print("    WARN: no liq data matched")

    # Test 2: Spot-checks at T = [1000, 5000, 50000]
    print("  Test 2: Spot-checks at test points...")
    for T in CAUSALITY_TEST_POINTS:
        # Liq event date < bar date
        if pd.notna(v3["_liq_event_date"].iloc[T]):
            bar_date = v3["_merge_date"].iloc[T]
            event_date = v3["_liq_event_date"].iloc[T]
            if event_date >= bar_date:
                print(
                    f"    FAIL at T={T}: liq event "
                    f"{str(event_date)[:10]} >= bar {str(bar_date)[:10]}"
                )
                causality_pass = False

        # USD conversion consistency
        if pd.notna(v3["liq_long_usd"].iloc[T]):
            expected = v3["liq_long_btc"].iloc[T] * v3["close"].iloc[T]
            actual = v3["liq_long_usd"].iloc[T]
            if abs(expected - actual) > 1e-6:
                print(f"    FAIL at T={T}: USD conversion mismatch")
                causality_pass = False

        # OI/CVD: values are from exact timestamp join -- causal by
        # construction (each bar's value depends only on its timestamp,
        # not on surrounding rows)

    if causality_pass:
        print("    PASS: all spot-checks clean")

    # Test 3: OI derived feature warmup check
    print("  Test 3: OI derived feature warmup check...")
    oi_start_idx = v3["oi_btc"].first_valid_index()
    if oi_start_idx is not None:
        oi_change_start = v3["oi_change_1h"].first_valid_index()
        if oi_change_start is not None:
            gap = oi_change_start - oi_start_idx
            print(
                f"    oi_change_1h starts {gap} bars after "
                f"OI data begins (expected >= 12)"
            )
            if gap < 12:
                print("    WARN: gap < 12, download may use min_periods")
        else:
            print("    WARN: oi_change_1h all NaN")
    else:
        print("    WARN: oi_btc all NaN")

    # Test 4: No duplicate columns in v3
    print("  Test 4: Column uniqueness...")
    dupes = v3.columns[v3.columns.duplicated()].tolist()
    if dupes:
        print(f"    FAIL: duplicate columns: {dupes}")
        causality_pass = False
    else:
        print(f"    PASS: all {len(v3.columns)} columns unique")

    # Drop temporary merge columns
    v3 = v3.drop(columns=["_merge_date", "_liq_event_date"])

    if causality_pass:
        print()
        print("  CAUSALITY GATE: ALL PASS")
    else:
        print()
        print("  CAUSALITY GATE: FAIL -- halting, v3 not written")
        sys.exit(1)
    print()

    # -------------------------------------------------------------------
    # Phase 7: Validation
    # -------------------------------------------------------------------
    print("Phase 7: Validation...")

    v3_rows = len(v3)
    v3_cols_count = len(v3.columns)
    new_col_count = v3_cols_count - v2_cols

    assert v3_rows == EXPECTED_V2_ROWS, (
        f"v3 row count {v3_rows} != expected {EXPECTED_V2_ROWS}"
    )
    print(f"  v3 rows: {v3_rows:,} (matches v2: OK)")
    print(
        f"  v3 cols: {v3_cols_count} "
        f"(v2: {v2_cols} + {new_col_count} new)"
    )

    # v2 modification time unchanged
    v2_mtime_after = os.path.getmtime(V2_PATH)
    if v2_mtime_after != v2_mtime:
        print(
            f"  WARNING: v2 mtime changed! "
            f"{v2_mtime:.6f} -> {v2_mtime_after:.6f}"
        )
    else:
        print(f"  v2 mtime unchanged: OK")

    # NaN rates for new columns
    print()
    print(f"  NaN rates for {new_col_count} new columns:")
    print(f"  {'Column':30s}  {'NaN':>8s}  {'Rate':>6s}  {'Non-NaN':>9s}")
    print(f"  {'-' * 30}  {'-' * 8}  {'-' * 6}  {'-' * 9}")
    for col in all_new_cols:
        n_nan = v3[col].isna().sum()
        rate = n_nan / v3_rows
        n_valid = v3_rows - n_nan
        print(
            f"  {col:30s}  {n_nan:>8,}  {rate:>5.1%}  {n_valid:>9,}"
        )
    print()

    # Free v2 reference
    del v2

    # -------------------------------------------------------------------
    # Phase 8: Write v3 parquet
    # -------------------------------------------------------------------
    print("Phase 8: Writing v3 parquet...")

    v3.to_parquet(V3_PATH, engine="pyarrow", compression="zstd", index=False)
    v3_size = os.path.getsize(V3_PATH)
    print(f"  Path: {V3_PATH}")
    print(f"  Size: {v3_size / 1e6:.1f} MB")
    print()

    # -------------------------------------------------------------------
    # Phase 9: Write feature catalog v3
    # -------------------------------------------------------------------
    print("Phase 9: Writing feature catalog v3...")

    write_feature_catalog_v3(
        CATALOG_V2, CATALOG_V3,
        oi_cols, cvd_cols, liq_btc_cols, liq_usd_cols,
    )
    print()

    # -------------------------------------------------------------------
    # Phase 10: Final verification (re-read v3)
    # -------------------------------------------------------------------
    print("Phase 10: Final verification (re-read v3)...")

    v3_check = pd.read_parquet(V3_PATH, columns=["bar_start_ts_utc"])
    assert len(v3_check) == EXPECTED_V2_ROWS, (
        f"v3 re-read: {len(v3_check)} rows vs expected {EXPECTED_V2_ROWS}"
    )
    print(f"  v3 re-read: {len(v3_check):,} rows -- OK")

    v2_mtime_final = os.path.getmtime(V2_PATH)
    v2_size_final = os.path.getsize(V2_PATH)
    assert v2_mtime_final == v2_mtime, "v2 mtime changed!"
    assert v2_size_final == v2_size, "v2 size changed!"
    print(f"  v2 unchanged: OK")
    print()

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    elapsed = time.time() - t0

    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print()
    print(f"Source          : {os.path.basename(V2_PATH)}")
    print(f"Output          : {os.path.basename(V3_PATH)}")
    print(f"v2 rows         : {v2_rows:,}")
    print(f"v3 rows         : {v3_rows:,}")
    print(f"v2 columns      : {v2_cols}")
    print(f"v3 columns      : {v3_cols_count}")
    print(f"New columns     : {new_col_count}")
    print(f"v3 file size    : {v3_size / 1e6:.1f} MB")
    print(f"Elapsed         : {elapsed:.0f}s")
    print()
    print("New column groups:")
    print(f"  OI      ({len(oi_cols):2d}): {', '.join(oi_cols)}")
    print(f"  CVD     ({len(cvd_cols):2d}): {', '.join(cvd_cols)}")
    print(f"  Liq BTC ({len(liq_btc_cols):2d}): {', '.join(liq_btc_cols)}")
    print(f"  Liq USD ({len(liq_usd_cols):2d}): {', '.join(liq_usd_cols)}")
    print()

    # Sample rows at bar 300000 (should have OI, CVD, and liq data)
    print("Sample: 3 rows at bar 300000 (new columns only):")
    print("-" * 60)
    sample = v3.iloc[300000:300003]
    for col in all_new_cols:
        vals = []
        for v in sample[col]:
            if pd.isna(v):
                vals.append("NaN")
            elif isinstance(v, (int, np.integer)):
                vals.append(str(v))
            else:
                vals.append(f"{v:.4f}")
        print(f"  {col:30s}: {vals}")
    print()

    # Liq causality demo: show bar date vs liq event date for 3 bars
    print("Liq causality demo (bar 300000):")
    bar_ts = v3["bar_start_ts_utc"].iloc[300000]
    bar_date = pd.Timestamp(bar_ts).normalize()
    liq_val = v3["liq_total_btc"].iloc[300000]
    print(f"  Bar timestamp : {bar_ts}")
    print(f"  Bar date      : {str(bar_date)[:10]}")
    print(f"  Liq total BTC : {liq_val:.4f}" if pd.notna(liq_val) else
          f"  Liq total BTC : NaN")
    print(f"  (liq data is from {str(bar_date - pd.Timedelta(days=1))[:10]})")
    print()

    print("=" * 60)
    print("DONE")
    print("=" * 60)

    log.info(
        "Completed: %d rows, %d->%d cols (+%d new), %.1f MB, %.0fs",
        v3_rows, v2_cols, v3_cols_count, new_col_count,
        v3_size / 1e6, elapsed,
    )


if __name__ == "__main__":
    main()
