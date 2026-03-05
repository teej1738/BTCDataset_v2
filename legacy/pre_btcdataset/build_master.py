"""
Script 3 — build_master.py
Multi-Timeframe Master Dataset Builder

Reads all enriched CSVs and builds a single master dataset by embedding
higher timeframe (HTF) context onto every 5m row — point-in-time safe
(no lookahead: only uses HTF candles already closed at each 5m timestamp).

Output:
  BTCUSDT_MASTER_2017-08-17_to_2026-03-01.csv
  BTCUSDT_MASTER_2017-08-17_to_2026-03-01.parquet

Requires: pip install pandas numpy pyarrow
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timezone

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

DATA_DIR   = r"C:\Users\tjall\Desktop\Trading\data"
OUTPUT_DIR = r"C:\Users\tjall\Desktop\Trading\data"

SYMBOL     = "BTCUSDT"
START_DATE = "2017-08-17"
END_DATE   = "2026-03-01"

# Base timeframe — HTF columns will be embedded onto every row of this
BASE_TF = "5m"

# Higher timeframes to embed (in order, smallest to largest)
HTF_LIST = ["15m", "30m", "1h", "4h", "1d"]

# HTF columns to embed onto base timeframe rows
# OHLCV + key ICT features from each HTF
HTF_COLUMNS = [
    # Core OHLCV
    "open", "high", "low", "close", "volume",
    # Volume / momentum
    "delta", "cvd", "atr_14",
    # Futures data
    "funding_rate", "open_interest",
    # Session / reference
    "session", "midnight_open", "ny_open_830",
    "pdh", "pdl", "pdc",
    "pwh", "pwl", "pwc",
    # Market structure
    "market_trend", "bos", "choch",
    # Swings
    "swing_high", "swing_low",
    "swing_high_price", "swing_low_price",
    # FVG
    "fvg_bull", "fvg_bear",
    "fvg_bull_top", "fvg_bull_bot", "fvg_bull_mid", "fvg_bull_mitigated",
    "fvg_bear_top", "fvg_bear_bot", "fvg_bear_mid", "fvg_bear_mitigated",
    # Order Blocks
    "ob_bull", "ob_bear",
    "ob_bull_top", "ob_bull_bot", "ob_bull_mitigated",
    "ob_bear_top", "ob_bear_bot", "ob_bear_mitigated",
    # Dealing range
    "dr_high", "dr_low", "dr_eq", "premium", "discount", "ote_zone",
    # Liquidity
    "bull_liq_sweep", "bear_liq_sweep",
    # New v2 features
    "displacement_flag", "candles_since_choch",
    "volume_percentile", "session_vwap", "price_vs_vwap",
    "fvg_bull_confluence", "fvg_bear_confluence",
    "ob_bull_confluence", "ob_bear_confluence",
    "liquidity_above", "liquidity_below",
]

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def enriched_path(interval):
    return os.path.join(
        DATA_DIR,
        f"{SYMBOL}_BINANCE_{interval}_{START_DATE}_to_{END_DATE}_enriched_v2.csv"
    )

def load_enriched(interval):
    path = enriched_path(interval)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Enriched file not found: {path}\nRun enrich_ict.py first.")
    print(f"  Loading {interval}...")
    df = pd.read_csv(path, low_memory=False)
    df["open_time_utc"] = pd.to_datetime(df["open_time_utc"], utc=True)
    df["open_time"]     = df["open_time"].astype(np.int64)
    df["close_time"]    = df["close_time"].astype(np.int64)
    df = df.sort_values("open_time").reset_index(drop=True)
    print(f"    {len(df):,} rows, {len(df.columns)} columns")
    return df

# ══════════════════════════════════════════════════════════════════════════════
# POINT-IN-TIME SAFE HTF MERGE
# ══════════════════════════════════════════════════════════════════════════════

def merge_htf(base_df, htf_df, htf_label):
    """
    Merge HTF data onto base timeframe using close_time of HTF candles.
    
    Key: we use close_time (not open_time) of HTF candles as the join key.
    This ensures a 1h candle that closes at 14:00 only becomes visible
    on the base timeframe AFTER 14:00 — no lookahead.
    
    Each HTF column is prefixed: e.g. 'close' becomes 'h1_close'
    """
    print(f"  Merging {htf_label}...")

    # Select only the columns we want from the HTF
    available_cols = [c for c in HTF_COLUMNS if c in htf_df.columns]
    htf_subset = htf_df[["close_time"] + available_cols].copy()

    # Rename columns with prefix
    prefix = htf_label.replace("m", "m").replace("h", "h").replace("d", "d")
    # Clean prefix: 5m->m5, 15m->m15, 1h->h1, 4h->h4, 1d->d1
    label_map = {
        "15m": "m15", "30m": "m30", "1h": "h1", "4h": "h4", "1d": "d1"
    }
    prefix = label_map.get(htf_label, htf_label)

    rename_map = {col: f"{prefix}_{col}" for col in available_cols}
    htf_subset = htf_subset.rename(columns=rename_map)

    # Use close_time of HTF as the join key — rename to open_time for merge_asof
    htf_subset = htf_subset.rename(columns={"close_time": "htf_close_time"})

    # merge_asof: for each base row, find the most recent HTF row
    # whose close_time <= base open_time (candle already closed)
    merged = pd.merge_asof(
        base_df.sort_values("open_time"),
        htf_subset.sort_values("htf_close_time"),
        left_on="open_time",
        right_on="htf_close_time",
        direction="backward"
    )

    merged.drop(columns=["htf_close_time"], inplace=True)
    merged = merged.sort_values("open_time").reset_index(drop=True)

    print(f"    Added {len(rename_map)} columns with prefix '{prefix}_'")
    return merged

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Building Master Dataset")
    print(f"  Base: {BASE_TF} | HTF: {HTF_LIST}")
    print(f"{'='*60}\n")

    # Load base timeframe
    print(f"Loading base timeframe ({BASE_TF})...")
    base_df = load_enriched(BASE_TF)
    print(f"  Base: {len(base_df):,} rows\n")

    # Load and merge each HTF
    for htf in HTF_LIST:
        print(f"Processing {htf}...")
        htf_df = load_enriched(htf)
        base_df = merge_htf(base_df, htf_df, htf)
        print(f"  Master now has {len(base_df.columns)} columns\n")
        del htf_df  # free memory

    # HTF confluence score: count of HTFs where market_trend == 1 (bullish) minus bearish
    # Range: -5 (all HTFs bearish) to +5 (all HTFs bullish)
    trend_cols = [c for c in base_df.columns if c.endswith("_market_trend") and not c == "market_trend"]
    if trend_cols:
        base_df["htf_confluence_score"] = sum(
            base_df[c].fillna(0) for c in trend_cols
        ).astype(np.int8)
        print(f"  htf_confluence_score computed from: {trend_cols}")

    # Final sort and dedup
    base_df = base_df.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)

    print(f"\nFinal master dataset:")
    print(f"  Rows:    {len(base_df):,}")
    print(f"  Columns: {len(base_df.columns)}")
    print(f"  Range:   {base_df['open_time_utc'].min()} -> {base_df['open_time_utc'].max()}")

    # Save CSV
    csv_name = f"{SYMBOL}_MASTER_{START_DATE}_to_{END_DATE}.csv"
    csv_path = os.path.join(OUTPUT_DIR, csv_name)
    print(f"\nSaving CSV -> {csv_name}...")
    base_df.to_csv(csv_path, index=False)
    csv_size = os.path.getsize(csv_path) / (1024 * 1024)
    print(f"  Done. Size: {csv_size:.1f} MB")

    # Save Parquet
    pq_name = f"{SYMBOL}_MASTER_{START_DATE}_to_{END_DATE}.parquet"
    pq_path = os.path.join(OUTPUT_DIR, pq_name)
    print(f"\nSaving Parquet -> {pq_name}...")
    base_df.to_parquet(pq_path, index=False, engine="pyarrow")
    pq_size = os.path.getsize(pq_path) / (1024 * 1024)
    print(f"  Done. Size: {pq_size:.1f} MB")

    print(f"\nParquet is {csv_size/pq_size:.1f}x smaller than CSV and loads faster.")

    # Write metadata
    meta = {
        "symbol":        SYMBOL,
        "base_tf":       BASE_TF,
        "htf_list":      HTF_LIST,
        "built_at":      datetime.now(timezone.utc).isoformat(),
        "start":         str(base_df["open_time_utc"].min()),
        "end":           str(base_df["open_time_utc"].max()),
        "total_rows":    len(base_df),
        "total_columns": len(base_df.columns),
        "columns":       list(base_df.columns),
        "csv_mb":        round(csv_size, 2),
        "parquet_mb":    round(pq_size, 2),
    }
    meta_path = os.path.join(OUTPUT_DIR, f"{SYMBOL}_MASTER_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata saved -> {SYMBOL}_MASTER_metadata.json")

    print(f"\n{'='*60}")
    print(f"  Master dataset complete.")
    print(f"  Load with: pd.read_parquet('{pq_name}')")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
