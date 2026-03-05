"""
Script 5 -- generate_labels_v2.py
Forward-Looking Label Generator (v2 -- triple-barrier, symmetric)

Generates R-multiple triple-barrier labels per D08 in STRATEGY_LOG:
  label_{long,short}_hit_{1,2,3}r_{12,24,48,96,288}c

Plus forward returns and max excursions for analysis:
  label_fwd_return_{N}c
  label_max_up_pct_{N}c
  label_max_down_pct_{N}c

Entry:     close of signal bar
Stop:      1 x ATR(14) adverse
Target:    R x ATR(14) favorable  (R = 1, 2, 3)
Horizon:   N candles  (12=1h, 24=2h, 48=4h, 96=8h, 288=24h on 5m bars)
Tie-break: if SL and TP both breached on the same candle, SL wins (conservative)

Fully vectorized with numpy sliding windows -- no Numba required.

Input:
  data/master/BTCUSDT_MASTER.parquet

Output:
  data/labeled/BTCUSDT_MASTER_labeled.parquet
  data/labeled/BTCUSDT_MASTER_labeled_metadata.json

Usage:
  python generate_labels_v2.py

Requires: pip install pandas pyarrow
"""

import pandas as pd
import numpy as np
import os
import json
import logging
import time
from datetime import datetime, timezone
from numpy.lib.stride_tricks import sliding_window_view

# ==============================================================================
# CONFIG
# ==============================================================================

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR  = os.path.join(BASE_DIR, "data", "master")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "labeled")
LOG_DIR    = os.path.join(BASE_DIR, "logs")

SYMBOL   = "BTCUSDT"

# Forward horizons in 5m candles
HORIZONS = [12, 24, 48, 96, 288]   # 1h, 2h, 4h, 8h, 24h

# R-multiples: risk 1 ATR, reward R * ATR
R_MULTIPLES = np.array([1.0, 2.0, 3.0])

ATR_COL = "ict_atr_14"


# ==============================================================================
# LOGGING
# ==============================================================================

def setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"generate_labels_{stamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return log_path


# ==============================================================================
# VECTORISED LABEL ENGINE
# ==============================================================================

def _first_true(mask):
    """
    For a 2-D boolean array (rows, window), return the index of the first
    True in each row.  Rows with no True get (window_size + 1) so they
    sort after everything.
    """
    hit_any = mask.any(axis=1)
    idx = np.argmax(mask, axis=1).astype(np.int32)
    idx[~hit_any] = mask.shape[1] + 1
    return idx


def generate_labels(df):
    """
    Generate all forward-looking labels.  Pure numpy, no Python row-loop.

    For each horizon N:
      1. Forward return  (close-to-close at bar i+N)
      2. Max up / down % (MFE / MAE)
      3. R-multiple triple-barrier: long and short, R = 1, 2, 3
    """
    n = len(df)
    close = df["close"].values.astype(np.float64)
    high  = df["high"].values.astype(np.float64)
    low   = df["low"].values.astype(np.float64)
    atr   = df[ATR_COL].values.astype(np.float64)

    for horizon in HORIZONS:
        t0 = time.perf_counter()
        tag = f"{horizon}c"
        valid = n - horizon          # rows with enough forward data

        logging.info(
            f"  Horizon {horizon} ({horizon * 5 / 60:.0f}h) "
            f"-- {valid:,} labelable rows ..."
        )

        # Forward-looking windows: for bar i -> bars i+1 .. i+horizon
        high_fwd = sliding_window_view(high[1:], horizon)[:valid]   # (valid, H)
        low_fwd  = sliding_window_view(low[1:],  horizon)[:valid]

        entry = close[:valid]
        atr_v = atr[:valid]
        invalid_atr = np.isnan(atr_v) | (atr_v <= 0)

        # -- Forward return (close-to-close) --
        fwd_ret = np.full(n, np.nan, dtype=np.float32)
        fwd_ret[:valid] = (
            (close[horizon : horizon + valid] - entry) / entry * 100.0
        ).astype(np.float32)
        df[f"label_fwd_return_{tag}"] = fwd_ret

        # -- Max excursions (MFE / MAE) --
        max_up = np.full(n, np.nan, dtype=np.float32)
        max_dn = np.full(n, np.nan, dtype=np.float32)
        max_up[:valid] = (
            (high_fwd.max(axis=1) - entry) / entry * 100.0
        ).astype(np.float32)
        max_dn[:valid] = (
            (entry - low_fwd.min(axis=1)) / entry * 100.0
        ).astype(np.float32)
        df[f"label_max_up_pct_{tag}"]   = max_up
        df[f"label_max_down_pct_{tag}"] = max_dn

        # -- Long stop-loss: entry - 1*ATR  (first bar where low <= SL) --
        sl_long_level = entry - atr_v
        sl_mask = low_fwd <= sl_long_level[:, None]
        first_sl_long = _first_true(sl_mask)
        del sl_mask

        # -- Short stop-loss: entry + 1*ATR  (first bar where high >= SL) --
        sl_short_level = entry + atr_v
        sl_mask = high_fwd >= sl_short_level[:, None]
        first_sl_short = _first_true(sl_mask)
        del sl_mask

        # -- For each R-multiple, compute TP hit index and compare to SL --
        for r_mult in R_MULTIPLES:
            ri = int(r_mult)

            # Long TP: entry + R*ATR   (first bar where high >= TP)
            tp_level = entry + r_mult * atr_v
            tp_mask  = high_fwd >= tp_level[:, None]
            first_tp = _first_true(tp_mask)
            del tp_mask

            arr = np.full(n, np.nan, dtype=np.float32)
            result = (first_tp < first_sl_long).astype(np.float32)
            result[invalid_atr] = np.nan
            arr[:valid] = result
            df[f"label_long_hit_{ri}r_{tag}"] = arr

            wr = np.nanmean(arr)
            logging.info(f"    {ri}R long  win-rate: {wr:.3f}")

            # Short TP: entry - R*ATR  (first bar where low <= TP)
            tp_level = entry - r_mult * atr_v
            tp_mask  = low_fwd <= tp_level[:, None]
            first_tp = _first_true(tp_mask)
            del tp_mask

            arr = np.full(n, np.nan, dtype=np.float32)
            result = (first_tp < first_sl_short).astype(np.float32)
            result[invalid_atr] = np.nan
            arr[:valid] = result
            df[f"label_short_hit_{ri}r_{tag}"] = arr

            wr = np.nanmean(arr)
            logging.info(f"    {ri}R short win-rate: {wr:.3f}")

        del first_sl_long, first_sl_short
        elapsed = time.perf_counter() - t0
        logging.info(f"    done in {elapsed:.1f}s")

    return df


# ==============================================================================
# QA REPORT
# ==============================================================================

def qa_report(df, original_cols):
    label_cols = sorted(c for c in df.columns if c.startswith("label_"))
    logging.info(f"\n{'-' * 60}")
    logging.info("QA REPORT")
    logging.info(f"{'-' * 60}")
    logging.info(f"  Rows:          {len(df):,}")
    logging.info(f"  Original cols: {original_cols}")
    logging.info(f"  Label cols:    {len(label_cols)}")
    logging.info(f"  Total cols:    {len(df.columns)}")

    # Win-rate table
    logging.info(f"\n  {'':4s} {'Horizon':>8s}  {'1R long':>8s} {'1R short':>9s}"
                 f"  {'2R long':>8s} {'2R short':>9s}"
                 f"  {'3R long':>8s} {'3R short':>9s}")
    logging.info(f"  {'':4s} {'-'*68}")

    for h in HORIZONS:
        tag = f"{h}c"
        hrs = h * 5 / 60
        parts = [f"  {hrs:4.0f}h {tag:>5s}"]
        for ri in [1, 2, 3]:
            lw = df[f"label_long_hit_{ri}r_{tag}"].mean()
            sw = df[f"label_short_hit_{ri}r_{tag}"].mean()
            parts.append(f"  {lw:>7.1%} {sw:>8.1%}")
        logging.info("".join(parts))

    # Forward return stats
    logging.info(f"\n  Forward return stats (%):")
    for h in HORIZONS:
        tag = f"{h}c"
        col = df[f"label_fwd_return_{tag}"]
        logging.info(
            f"    {tag:>5s}:  mean={col.mean():+.3f}  "
            f"std={col.std():.3f}  "
            f"median={col.median():+.3f}  "
            f"[{col.min():.1f}, {col.max():.1f}]"
        )

    # NaN counts
    logging.info(f"\n  NaN rows per horizon (tail + ATR warmup):")
    for h in HORIZONS:
        tag = f"{h}c"
        nan_count = df[f"label_long_hit_1r_{tag}"].isna().sum()
        logging.info(f"    {tag:>5s}:  {nan_count:,}  ({nan_count / len(df) * 100:.2f}%)")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    log_path = setup_logging()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logging.info(f"\n{'=' * 60}")
    logging.info(f"  Label Generation  (v2 -- triple-barrier)")
    logging.info(f"  Horizons: {HORIZONS}  |  R-multiples: {list(R_MULTIPLES)}")
    logging.info(f"  SL = 1 x {ATR_COL}  |  TP = R x {ATR_COL}")
    logging.info(f"{'=' * 60}\n")

    # -- Load --
    master_path = os.path.join(INPUT_DIR, f"{SYMBOL}_MASTER.parquet")
    if not os.path.exists(master_path):
        logging.error(f"Master parquet not found: {master_path}")
        logging.error("Run build_master_v4.py first.")
        return

    logging.info("Loading master dataset ...")
    df = pd.read_parquet(master_path)
    df = df.sort_values("bar_start_ts_utc").reset_index(drop=True)
    original_cols = len(df.columns)
    logging.info(f"  {len(df):,} rows x {original_cols} columns\n")

    # -- Generate --
    logging.info("Generating labels ...")
    t0 = time.perf_counter()
    df = generate_labels(df)
    elapsed = time.perf_counter() - t0
    logging.info(f"\n  All labels generated in {elapsed:.1f}s")

    # -- QA --
    qa_report(df, original_cols)

    # -- Save --
    out_path = os.path.join(OUTPUT_DIR, f"{SYMBOL}_MASTER_labeled.parquet")
    logging.info(f"\nSaving -> {out_path}")
    df.to_parquet(out_path, index=False, engine="pyarrow")
    size_mb = os.path.getsize(out_path) / 1024**2
    logging.info(f"  Done. Size: {size_mb:.1f} MB")

    # -- Metadata --
    label_cols = sorted(c for c in df.columns if c.startswith("label_"))
    meta = {
        "symbol":         SYMBOL,
        "horizons":       HORIZONS,
        "r_multiples":    [int(r) for r in R_MULTIPLES],
        "atr_column":     ATR_COL,
        "sl_mult":        1.0,
        "tie_break":      "SL wins on same candle (conservative)",
        "built_at":       datetime.now(timezone.utc).isoformat(),
        "start":          str(df["bar_start_ts_utc"].min()),
        "end":            str(df["bar_start_ts_utc"].max()),
        "total_rows":     len(df),
        "total_columns":  len(df.columns),
        "label_columns":  label_cols,
        "label_count":    len(label_cols),
        "elapsed_sec":    round(elapsed, 1),
        "parquet_mb":     round(size_mb, 2),
        "log_file":       os.path.basename(log_path),
    }
    meta_path = os.path.join(OUTPUT_DIR, f"{SYMBOL}_MASTER_labeled_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logging.info(f"  Metadata -> {meta_path}")

    logging.info(f"\n{'=' * 60}")
    logging.info(f"  Labels complete.")
    logging.info(f"  {len(df):,} rows x {len(df.columns)} columns")
    logging.info(f"  {len(label_cols)} label columns  |  {size_mb:.1f} MB")
    logging.info(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
