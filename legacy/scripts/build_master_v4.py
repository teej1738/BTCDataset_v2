"""
Script 4 — build_master_v4.py
Multi-Timeframe Master Dataset Builder (v4 — parquet, v2 column conventions)

Reads enriched parquet files and builds a single master dataset by embedding
higher-timeframe (HTF) context onto every perp 5m row — point-in-time safe
(no lookahead: only uses HTF candles already closed at each 5m timestamp).

Inputs:
  data/enriched/BTCUSDT_perp_{5m,15m,30m,1h,4h,1d}_enriched.parquet
  data/enriched/BTCUSDT_spot_1d_enriched.parquet  (D01 spot trend)

Output:
  data/master/BTCUSDT_MASTER.parquet
  data/master/BTCUSDT_MASTER_metadata.json

Usage:
  python build_master_v4.py

Requires: pip install pandas pyarrow
"""

import pandas as pd
import numpy as np
import os
import json
import logging
from datetime import datetime, timezone

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR  = os.path.join(BASE_DIR, "data", "enriched")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "master")
LOG_DIR    = os.path.join(BASE_DIR, "logs")

SYMBOL   = "BTCUSDT"
BASE_TF  = "5m"
HTF_LIST = ["15m", "30m", "1h", "4h", "1d"]

HTF_PREFIX_MAP = {
    "15m": "m15", "30m": "m30", "1h": "h1", "4h": "h4", "1d": "d1",
}

# Curated columns to embed from each HTF (~49 per HTF)
HTF_COLUMNS = [
    # Core price
    "open", "high", "low", "close", "volume_base",
    # Volatility / momentum
    "ict_atr_14", "ict_atr_ratio", "ict_realized_vol_20",
    "cvd_delta", "cvd_daily", "cvd_zscore",
    # Market structure
    "ict_market_trend", "ict_bos", "ict_choch",
    # Swings
    "ict_swing_high", "ict_swing_low",
    "ict_swing_high_price", "ict_swing_low_price",
    # FVG (multi-track)
    "ict_fvg_bull", "ict_fvg_bear",
    "ict_fvg_bull_count_total", "ict_fvg_bear_count_total",
    "ict_fvg_bull_in_zone", "ict_fvg_bear_in_zone",
    "ict_fvg_bull_nearest_top", "ict_fvg_bull_nearest_bot",
    "ict_fvg_bear_nearest_top", "ict_fvg_bear_nearest_bot",
    # Order blocks
    "ict_ob_bull", "ict_ob_bear",
    "ict_ob_bull_top", "ict_ob_bull_bot", "ict_ob_bull_mitigated",
    "ict_ob_bear_top", "ict_ob_bear_bot", "ict_ob_bear_mitigated",
    # Dealing range
    "ict_dr_high", "ict_dr_low", "ict_dr_eq",
    "ict_premium", "ict_discount", "ict_ote_zone",
    # Liquidity / displacement
    "ict_bull_liq_sweep", "ict_bear_liq_sweep", "ict_disp_any",
    # Session / volume
    "sess_vwap", "sess_price_vs_vwap", "ict_volume_pct",
    # Funding (perp-only)
    "fund_rate_zscore",
]


# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════

def setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"build_master_{stamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )
    return log_path


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def enriched_path(instrument, interval):
    return os.path.join(
        INPUT_DIR, f"{SYMBOL}_{instrument}_{interval}_enriched.parquet"
    )


def load_enriched(instrument, interval):
    path = enriched_path(instrument, interval)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Enriched file not found: {path}\nRun enrich_ict_v4.py first."
        )
    logging.info(f"  Loading {instrument} {interval} ...")
    df = pd.read_parquet(path)
    df = df.sort_values("bar_start_ts_utc").reset_index(drop=True)
    logging.info(f"    {len(df):,} rows, {len(df.columns)} columns")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# POINT-IN-TIME SAFE HTF MERGE
# ══════════════════════════════════════════════════════════════════════════════

def merge_htf(base_df, htf_df, htf_interval):
    """
    Merge HTF data onto base 5m using bar_end_ts_ms of HTF candles.

    merge_asof(direction='backward') ensures an HTF candle only appears on 5m
    rows whose bar_start_ts_ms >= HTF bar_end_ts_ms (candle already closed).
    No lookahead.
    """
    prefix = HTF_PREFIX_MAP[htf_interval]
    logging.info(f"  Merging {htf_interval} -> prefix '{prefix}_' ...")

    # Select curated columns that exist in this HTF
    available = [c for c in HTF_COLUMNS if c in htf_df.columns]
    missing = [c for c in HTF_COLUMNS if c not in htf_df.columns]
    if missing:
        logging.info(f"    Skipped {len(missing)} missing cols: {missing}")

    # Build subset: temp join key + curated columns
    # Use bar_end_ts_utc (datetime64[ms, UTC]) — bar_start_ts_ms is low-precision
    htf_subset = htf_df[["bar_end_ts_utc"] + available].copy()
    htf_subset = htf_subset.rename(columns={"bar_end_ts_utc": "_htf_close"})

    # Prefix curated columns
    rename_map = {col: f"{prefix}_{col}" for col in available}
    htf_subset = htf_subset.rename(columns=rename_map)
    htf_subset = htf_subset.sort_values("_htf_close").reset_index(drop=True)

    merged = pd.merge_asof(
        base_df,
        htf_subset,
        left_on="bar_start_ts_utc",
        right_on="_htf_close",
        direction="backward",
    )
    merged.drop(columns=["_htf_close"], inplace=True)

    logging.info(f"    Added {len(available)} columns with prefix '{prefix}_'")
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# QA
# ══════════════════════════════════════════════════════════════════════════════

def qa_report(df):
    """Log QA metrics for the final master dataset."""
    logging.info(f"\n{'-' * 60}")
    logging.info("QA REPORT")
    logging.info(f"{'-' * 60}")
    logging.info(f"  Rows:      {len(df):,}")
    logging.info(f"  Columns:   {len(df.columns)}")
    logging.info(
        f"  Date range: {df['bar_start_ts_utc'].min()}"
        f" -> {df['bar_start_ts_utc'].max()}"
    )

    # NaN % by HTF prefix
    logging.info("\n  NaN % by HTF prefix:")
    for pfx in list(HTF_PREFIX_MAP.values()) + ["htf"]:
        cols = [c for c in df.columns if c.startswith(f"{pfx}_")]
        if cols:
            nan_pct = df[cols].isna().mean().mean() * 100
            logging.info(f"    {pfx:6s}  {len(cols):3d} cols  {nan_pct:5.1f}% NaN")

    # Confluence score distribution
    if "htf_confluence_score" in df.columns:
        logging.info("\n  htf_confluence_score distribution:")
        dist = df["htf_confluence_score"].value_counts().sort_index()
        for val, cnt in dist.items():
            logging.info(f"    {val:+d}: {cnt:>8,} ({cnt / len(df) * 100:.1f}%)")
        rng = df["htf_confluence_score"]
        logging.info(f"    Range: [{rng.min()}, {rng.max()}]")

    # Spot trend
    if "htf_d1_spot_trend" in df.columns:
        nan_pct = df["htf_d1_spot_trend"].isna().mean() * 100
        logging.info(f"\n  htf_d1_spot_trend: {nan_pct:.1f}% NaN")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    log_path = setup_logging()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logging.info(f"\n{'=' * 60}")
    logging.info(f"  Building Master Dataset  (v4)")
    logging.info(f"  Base: perp {BASE_TF}  |  HTF: {HTF_LIST}")
    logging.info(f"{'=' * 60}\n")

    # -- 1. Load base timeframe --------------------------------------------
    logging.info("Loading base timeframe (perp 5m) ...")
    base_df = load_enriched("perp", BASE_TF)
    base_rows = len(base_df)
    logging.info(f"  Base: {base_rows:,} rows\n")

    # -- 2. Merge each perp HTF -------------------------------------------
    for htf in HTF_LIST:
        logging.info(f"Processing perp {htf} ...")
        htf_df = load_enriched("perp", htf)
        base_df = merge_htf(base_df, htf_df, htf)
        logging.info(f"  Master now: {len(base_df.columns)} columns\n")
        del htf_df

    # -- 3. Spot D1 trend (D01 bonus feature) -----------------------------
    logging.info("Merging spot D1 trend ...")
    spot_1d = load_enriched("spot", "1d")

    spot_subset = spot_1d[["bar_end_ts_utc", "ict_market_trend"]].copy()
    spot_subset = spot_subset.rename(columns={
        "bar_end_ts_utc": "_htf_close",
        "ict_market_trend": "htf_d1_spot_trend",
    })
    spot_subset = spot_subset.sort_values("_htf_close").reset_index(drop=True)

    base_df = pd.merge_asof(
        base_df,
        spot_subset,
        left_on="bar_start_ts_utc",
        right_on="_htf_close",
        direction="backward",
    )
    base_df.drop(columns=["_htf_close"], inplace=True)
    del spot_1d, spot_subset
    logging.info("  Added htf_d1_spot_trend\n")

    # -- 4. Derived: HTF confluence score ---------------------------------
    trend_cols = [f"{pfx}_ict_market_trend" for pfx in HTF_PREFIX_MAP.values()]
    existing_trend = [c for c in trend_cols if c in base_df.columns]
    base_df["htf_confluence_score"] = sum(
        base_df[c].fillna(0) for c in existing_trend
    ).astype(np.int8)
    logging.info(f"  htf_confluence_score from {existing_trend}")

    # -- 5. Final sort & dedup safety -------------------------------------
    before = len(base_df)
    base_df = (
        base_df
        .sort_values("bar_start_ts_utc")
        .drop_duplicates("bar_start_ts_utc")
        .reset_index(drop=True)
    )
    after = len(base_df)
    if before != after:
        logging.warning(f"  Dropped {before - after} duplicate rows")

    assert len(base_df) == base_rows, (
        f"Row count changed: {base_rows:,} -> {len(base_df):,}"
    )

    # -- 6. QA ------------------------------------------------------------
    qa_report(base_df)

    # -- 7. Save parquet --------------------------------------------------
    pq_path = os.path.join(OUTPUT_DIR, f"{SYMBOL}_MASTER.parquet")
    logging.info(f"\nSaving parquet -> {pq_path}")
    base_df.to_parquet(pq_path, index=False, engine="pyarrow")
    pq_size = os.path.getsize(pq_path) / (1024 * 1024)
    logging.info(f"  Done. Size: {pq_size:.1f} MB")

    # -- 8. Metadata ------------------------------------------------------
    meta = {
        "symbol":              SYMBOL,
        "base_tf":             BASE_TF,
        "htf_list":            HTF_LIST,
        "htf_prefix_map":      HTF_PREFIX_MAP,
        "htf_columns_per_tf":  len(HTF_COLUMNS),
        "built_at":            datetime.now(timezone.utc).isoformat(),
        "start":               str(base_df["bar_start_ts_utc"].min()),
        "end":                 str(base_df["bar_start_ts_utc"].max()),
        "total_rows":          len(base_df),
        "total_columns":       len(base_df.columns),
        "parquet_mb":          round(pq_size, 2),
        "log_file":            os.path.basename(log_path),
        "columns":             list(base_df.columns),
    }
    meta_path = os.path.join(OUTPUT_DIR, f"{SYMBOL}_MASTER_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logging.info(f"  Metadata -> {meta_path}")

    logging.info(f"\n{'=' * 60}")
    logging.info(f"  Master dataset complete.")
    logging.info(f"  {len(base_df):,} rows × {len(base_df.columns)} columns")
    logging.info(f"  {pq_size:.1f} MB parquet")
    logging.info(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
