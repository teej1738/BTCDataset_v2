"""
generate_labels.py
Forward-looking label generation for BTCUSDT master dataset.

Adds outcome columns to the master parquet — these are the targets
for backtesting and ML training.

IMPORTANT: These columns look FORWARD in time. They must NEVER be used
as input features in any model. Only use them as prediction targets (y).
All feature columns (x) already exist in the dataset and are point-in-time safe.

Labels generated:
  --- Price outcome ---
  fwd_max_up_pct_{N}      max % price moved UP in next N candles (best case long)
  fwd_max_down_pct_{N}    max % price moved DOWN in next N candles (worst case long)
  fwd_return_{N}          close-to-close % return over next N candles
  fwd_direction_{N}       1 if fwd_return > 0, -1 if < 0, 0 if flat

  --- Risk/Reward outcome ---
  fwd_hit_tp_before_sl_{N}  1 if price hit TP (1.5R) before SL (1R) in next N candles
  fwd_rr_ratio_{N}          actual max_up / max_down ratio (realized RR)

  --- Structure outcome ---
  fwd_fvg_bull_filled_{N}   1 if current bull FVG top was touched within N candles
  fwd_fvg_bear_filled_{N}   1 if current bear FVG bot was touched within N candles
  fwd_ob_bull_tested_{N}    1 if current bull OB top was touched within N candles
  fwd_ob_bear_tested_{N}    1 if current bear OB bot was touched within N candles
  fwd_next_bos_direction    direction of the next BOS from this candle (1=bull, -1=bear)
  fwd_candles_to_next_bos   how many candles until next BOS

  --- Session/Time outcome ---
  fwd_killzone_range_{session}  high-low range of the next occurrence of each session

Horizons (N candles on 5m base = real time):
  12  = 1 hour
  24  = 2 hours
  48  = 4 hours
  96  = 8 hours
  288 = 24 hours (1 day)

Requires: pip install pandas numpy pyarrow
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timezone

# ==============================================================================
# CONFIG
# ==============================================================================

DATA_DIR   = r"C:\Users\tjall\Desktop\Trading\data"
SYMBOL     = "BTCUSDT"
START_DATE = "2017-08-17"
END_DATE   = "2026-03-01"

MASTER_PARQUET = os.path.join(DATA_DIR, f"{SYMBOL}_MASTER_{START_DATE}_to_{END_DATE}.parquet")
OUTPUT_PARQUET = os.path.join(DATA_DIR, f"{SYMBOL}_MASTER_{START_DATE}_to_{END_DATE}_labeled.parquet")
OUTPUT_CSV     = os.path.join(DATA_DIR, f"{SYMBOL}_MASTER_{START_DATE}_to_{END_DATE}_labeled.csv")

# Forward horizons in candles (5m base)
HORIZONS = [12, 24, 48, 96, 288]

# ATR multiplier for stop loss
SL_ATR_MULT = 1.0   # stop = 1x ATR below entry (long) / above entry (short)
TP_ATR_MULT = 1.5   # target = 1.5x ATR above entry (long) / below entry (short)

# ==============================================================================
# LOAD
# ==============================================================================

def load_master():
    print(f"Loading master parquet...")
    df = pd.read_parquet(MASTER_PARQUET)
    df["open_time_utc"] = pd.to_datetime(df["open_time_utc"], utc=True)
    df = df.sort_values("open_time").reset_index(drop=True)
    print(f"  Loaded {len(df):,} rows x {len(df.columns)} columns")
    return df

# ==============================================================================
# PRICE OUTCOME LABELS
# ==============================================================================

def add_price_outcomes(df):
    """
    For each horizon N, compute:
      - max high in next N candles (best case long exit)
      - min low in next N candles (worst case long / best case short exit)
      - close N candles forward
      - direction (1 = up, -1 = down)
    """
    print(f"  Computing price outcomes for horizons {HORIZONS}...")

    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values
    size   = len(df)

    for n in HORIZONS:
        print(f"    Horizon {n} candles ({n*5//60}h {n*5%60}m)...")

        max_up   = np.full(size, np.nan)
        max_down = np.full(size, np.nan)
        fwd_ret  = np.full(size, np.nan)

        for i in range(size - n):
            entry       = closes[i]
            future_high = np.max(highs[i+1 : i+n+1])
            future_low  = np.min(lows[i+1  : i+n+1])
            future_close= closes[i+n]

            max_up[i]   = (future_high  - entry) / entry * 100
            max_down[i] = (entry - future_low)   / entry * 100
            fwd_ret[i]  = (future_close - entry) / entry * 100

        df[f"fwd_max_up_pct_{n}"]   = max_up
        df[f"fwd_max_down_pct_{n}"] = max_down
        df[f"fwd_return_{n}"]       = fwd_ret
        df[f"fwd_direction_{n}"]    = np.sign(fwd_ret).astype(np.int8)

    return df

# ==============================================================================
# RISK/REWARD LABELS
# ==============================================================================

def add_rr_outcomes(df):
    """
    Simulate entering a trade at close with ATR-based SL and TP.
    For each horizon N:
      - Long: SL = close - atr*SL_MULT, TP = close + atr*TP_MULT
      - Short: SL = close + atr*SL_MULT, TP = close - atr*TP_MULT
      - Check if TP or SL hit first within N candles
      - fwd_hit_tp_before_sl_{N}: 1=TP hit first (long wins), 0=SL hit first, -1=neither
      - fwd_rr_realized_{N}: max_up / max_down (raw RR, direction-agnostic)
    """
    print(f"  Computing RR outcomes...")

    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values
    atr    = df["atr_14"].values
    size   = len(df)

    for n in HORIZONS:
        print(f"    RR horizon {n}...")
        tp_before_sl_long  = np.full(size, -1, dtype=np.int8)  # -1 = neither hit
        tp_before_sl_short = np.full(size, -1, dtype=np.int8)
        rr_realized        = np.full(size, np.nan)

        for i in range(size - n):
            if np.isnan(atr[i]):
                continue
            entry = closes[i]
            sl    = atr[i] * SL_ATR_MULT
            tp    = atr[i] * TP_ATR_MULT

            long_sl  = entry - sl
            long_tp  = entry + tp
            short_sl = entry + sl
            short_tp = entry - tp

            # Walk forward candle by candle to find first hit
            long_result  = -1
            short_result = -1

            for j in range(i+1, i+n+1):
                h = highs[j]
                l = lows[j]

                if long_result == -1:
                    if l <= long_sl:
                        long_result = 0   # SL hit
                    elif h >= long_tp:
                        long_result = 1   # TP hit

                if short_result == -1:
                    if h >= short_sl:
                        short_result = 0  # SL hit
                    elif l <= short_tp:
                        short_result = 1  # TP hit

                if long_result != -1 and short_result != -1:
                    break

            tp_before_sl_long[i]  = long_result
            tp_before_sl_short[i] = short_result

            # Raw RR: best move up / best move down (symmetrical measure)
            future_highs = highs[i+1:i+n+1]
            future_lows  = lows[i+1:i+n+1]
            best_up   = np.max(future_highs) - entry
            best_down = entry - np.min(future_lows)
            if best_down > 0:
                rr_realized[i] = best_up / best_down

        df[f"fwd_long_wins_{n}"]  = tp_before_sl_long
        df[f"fwd_short_wins_{n}"] = tp_before_sl_short
        df[f"fwd_rr_realized_{n}"]= rr_realized

    return df

# ==============================================================================
# STRUCTURE OUTCOME LABELS
# ==============================================================================

def add_structure_outcomes(df):
    """
    Did key ICT levels get tested within N candles?
    - FVG fill: price touches into gap
    - OB test: price touches into order block body
    - Next BOS: direction and distance of next market structure break
    """
    print(f"  Computing structure outcomes...")

    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values
    size   = len(df)

    # FVG fill labels
    fvg_bull_top = df["fvg_bull_top"].values if "fvg_bull_top" in df.columns else np.full(size, np.nan)
    fvg_bull_bot = df["fvg_bull_bot"].values if "fvg_bull_bot" in df.columns else np.full(size, np.nan)
    fvg_bear_top = df["fvg_bear_top"].values if "fvg_bear_top" in df.columns else np.full(size, np.nan)
    fvg_bear_bot = df["fvg_bear_bot"].values if "fvg_bear_bot" in df.columns else np.full(size, np.nan)

    ob_bull_top = df["ob_bull_top"].values if "ob_bull_top" in df.columns else np.full(size, np.nan)
    ob_bull_bot = df["ob_bull_bot"].values if "ob_bull_bot" in df.columns else np.full(size, np.nan)
    ob_bear_top = df["ob_bear_top"].values if "ob_bear_top" in df.columns else np.full(size, np.nan)
    ob_bear_bot = df["ob_bear_bot"].values if "ob_bear_bot" in df.columns else np.full(size, np.nan)

    bos = df["bos"].values if "bos" in df.columns else np.zeros(size)

    for n in [48, 96, 288]:
        print(f"    Structure horizon {n}...")

        fvg_bull_filled = np.zeros(size, dtype=np.int8)
        fvg_bear_filled = np.zeros(size, dtype=np.int8)
        ob_bull_tested  = np.zeros(size, dtype=np.int8)
        ob_bear_tested  = np.zeros(size, dtype=np.int8)

        for i in range(size - n):
            end = min(i + n + 1, size)

            # Bull FVG fill: price trades down into the gap (low touches gap)
            if not np.isnan(fvg_bull_top[i]) and not np.isnan(fvg_bull_bot[i]):
                for j in range(i+1, end):
                    if lows[j] <= fvg_bull_top[i]:
                        fvg_bull_filled[i] = 1
                        break

            # Bear FVG fill: price trades up into the gap (high touches gap)
            if not np.isnan(fvg_bear_top[i]) and not np.isnan(fvg_bear_bot[i]):
                for j in range(i+1, end):
                    if highs[j] >= fvg_bear_bot[i]:
                        fvg_bear_filled[i] = 1
                        break

            # Bull OB test: price trades down into OB body
            if not np.isnan(ob_bull_top[i]) and not np.isnan(ob_bull_bot[i]):
                for j in range(i+1, end):
                    if lows[j] <= ob_bull_top[i]:
                        ob_bull_tested[i] = 1
                        break

            # Bear OB test: price trades up into OB body
            if not np.isnan(ob_bear_top[i]) and not np.isnan(ob_bear_bot[i]):
                for j in range(i+1, end):
                    if highs[j] >= ob_bear_bot[i]:
                        ob_bear_tested[i] = 1
                        break

        df[f"fwd_fvg_bull_filled_{n}"] = fvg_bull_filled
        df[f"fwd_fvg_bear_filled_{n}"] = fvg_bear_filled
        df[f"fwd_ob_bull_tested_{n}"]  = ob_bull_tested
        df[f"fwd_ob_bear_tested_{n}"]  = ob_bear_tested

    # Next BOS direction and distance (no horizon limit — just find the next one)
    print(f"    Next BOS direction and distance...")
    next_bos_dir  = np.zeros(size, dtype=np.int8)
    candles_to_bos= np.full(size, np.nan)

    # Walk backward from end: for each candle, find next BOS ahead
    # Efficient: precompute next BOS index for each position
    next_bos_idx = np.full(size, -1, dtype=np.int64)
    last_bos = -1
    last_bos_dir = 0
    for i in range(size - 1, -1, -1):
        if bos[i] != 0:
            last_bos = i
            last_bos_dir = int(bos[i])
        next_bos_idx[i] = last_bos

    for i in range(size):
        nb = next_bos_idx[i]
        if nb > i:
            next_bos_dir[i]   = int(bos[nb])
            candles_to_bos[i] = nb - i

    df["fwd_next_bos_direction"]  = next_bos_dir
    df["fwd_candles_to_next_bos"] = candles_to_bos

    return df

# ==============================================================================
# COMPOSITE LABELS (useful for ML classification)
# ==============================================================================

def add_composite_labels(df):
    """
    Higher-level labels that combine multiple signals into clean targets.

    fwd_bias_{N}: 1=bullish (up more than down), -1=bearish, 0=balanced
    fwd_quality_{N}: 0-3 score of how clean the move was
      3 = strong directional move, low drawdown
      2 = decent move
      1 = choppy
      0 = no clear direction
    fwd_long_quality_{N}: specifically for long setups
      2 = hit TP before SL AND return positive
      1 = return positive but didn't cleanly hit TP
      0 = SL hit or negative return
    """
    print(f"  Computing composite labels...")

    for n in HORIZONS:
        if f"fwd_max_up_pct_{n}" not in df.columns:
            continue

        up   = df[f"fwd_max_up_pct_{n}"]
        down = df[f"fwd_max_down_pct_{n}"]
        ret  = df[f"fwd_return_{n}"]

        # Bias: which direction dominated
        df[f"fwd_bias_{n}"] = np.where(
            up > down * 1.5,  1,   # bull: up move was 1.5x larger
            np.where(down > up * 1.5, -1,  # bear: down move was 1.5x larger
            0)  # balanced/choppy
        ).astype(np.int8)

        # Long quality
        if f"fwd_long_wins_{n}" in df.columns:
            long_wins = df[f"fwd_long_wins_{n}"]
            df[f"fwd_long_quality_{n}"] = np.where(
                long_wins == 1,  2,   # TP hit before SL — clean long win
                np.where(ret > 0, 1,  # positive return but messy
                0)                    # loss
            ).astype(np.int8)

        # Short quality
        if f"fwd_short_wins_{n}" in df.columns:
            short_wins = df[f"fwd_short_wins_{n}"]
            df[f"fwd_short_quality_{n}"] = np.where(
                short_wins == 1,  2,
                np.where(ret < 0, 1,
                0)
            ).astype(np.int8)

    return df

# ==============================================================================
# SUMMARY STATS
# ==============================================================================

def print_label_summary(df):
    print(f"\n  Label summary (base rates — what % of candles are 'long wins'):")
    print(f"  {'Horizon':<12} {'Long wins%':<14} {'Short wins%':<14} {'Bull bias%':<12} {'Avg RR'}")
    print(f"  {'-'*65}")

    for n in HORIZONS:
        lw_col  = f"fwd_long_wins_{n}"
        sw_col  = f"fwd_short_wins_{n}"
        bi_col  = f"fwd_bias_{n}"
        rr_col  = f"fwd_rr_realized_{n}"

        lw  = df[lw_col].eq(1).mean()  * 100 if lw_col  in df.columns else 0
        sw  = df[sw_col].eq(1).mean()  * 100 if sw_col  in df.columns else 0
        bi  = df[bi_col].eq(1).mean()  * 100 if bi_col  in df.columns else 0
        rr  = df[rr_col].mean()               if rr_col in df.columns else 0
        hrs = n * 5 / 60

        print(f"  {n} ({hrs:.0f}h){'':<6} {lw:>6.1f}%{'':<8} {sw:>6.1f}%{'':<8} {bi:>6.1f}%{'':<6} {rr:.2f}")

    # BOS stats
    if "fwd_next_bos_direction" in df.columns:
        bull_bos = (df["fwd_next_bos_direction"] == 1).mean() * 100
        bear_bos = (df["fwd_next_bos_direction"] == -1).mean() * 100
        avg_dist = df["fwd_candles_to_next_bos"].mean()
        print(f"\n  Next BOS: {bull_bos:.1f}% bullish / {bear_bos:.1f}% bearish")
        print(f"  Avg candles to next BOS: {avg_dist:.1f} ({avg_dist*5/60:.1f}h)")

    # FVG fill rates
    for n in [96, 288]:
        col = f"fwd_fvg_bull_filled_{n}"
        if col in df.columns:
            rate = df[col].mean() * 100
            print(f"  Bull FVG fill rate within {n} candles ({n*5//60}h): {rate:.1f}%")
        col = f"fwd_fvg_bear_filled_{n}"
        if col in df.columns:
            rate = df[col].mean() * 100
            print(f"  Bear FVG fill rate within {n} candles ({n*5//60}h): {rate:.1f}%")

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print(f"\n{'='*60}")
    print(f"  Label Generation")
    print(f"  Symbol: {SYMBOL} | {START_DATE} to {END_DATE}")
    print(f"{'='*60}\n")

    if not os.path.exists(MASTER_PARQUET):
        print(f"ERROR: Master parquet not found: {MASTER_PARQUET}")
        print(f"Run build_master.py first.")
        return

    df = load_master()
    original_cols = len(df.columns)

    print(f"\n[1/4] Price outcome labels...")
    df = add_price_outcomes(df)

    print(f"\n[2/4] Risk/reward outcome labels...")
    df = add_rr_outcomes(df)

    print(f"\n[3/4] Structure outcome labels...")
    df = add_structure_outcomes(df)

    print(f"\n[4/4] Composite labels...")
    df = add_composite_labels(df)

    new_cols = len(df.columns) - original_cols
    print(f"\n  Added {new_cols} label columns")
    print(f"  Total columns: {len(df.columns)}")

    label_cols = [c for c in df.columns if c.startswith("fwd_")]
    print(f"  Label columns: {len(label_cols)}")

    print_label_summary(df)

    # Save — parquet only (labels make CSV enormous)
    print(f"\nSaving labeled parquet...")
    df.to_parquet(OUTPUT_PARQUET, index=False, engine="pyarrow")
    size_mb = os.path.getsize(OUTPUT_PARQUET) / 1024**2
    print(f"  Done. Size: {size_mb:.1f} MB -> {os.path.basename(OUTPUT_PARQUET)}")

    # Save label columns list for reference
    label_ref_path = os.path.join(DATA_DIR, "label_columns.txt")
    with open(label_ref_path, "w") as f:
        f.write("FORWARD-LOOKING LABEL COLUMNS\n")
        f.write("DO NOT USE AS FEATURES - TARGETS ONLY\n")
        f.write("="*50 + "\n\n")
        for col in sorted(label_cols):
            null_pct = df[col].isnull().mean() * 100
            f.write(f"{col:<45} {null_pct:.1f}% null\n")
    print(f"  Label reference saved -> label_columns.txt")

    print(f"\n{'='*60}")
    print(f"  Labels complete.")
    print(f"  Load with:")
    print(f"  df = pd.read_parquet('{os.path.basename(OUTPUT_PARQUET)}')")
    print(f"  feature_cols = [c for c in df.columns if not c.startswith('fwd_')]")
    print(f"  target_col   = 'fwd_long_quality_48'  # example target")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
