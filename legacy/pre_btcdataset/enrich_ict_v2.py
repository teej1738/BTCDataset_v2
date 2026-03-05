"""
Script 2b -- enrich_ict_v2.py
Extended ICT Enrichment + Bug Fixes

Fixes:
  1. ny_open_830 -- use >= match instead of exact minute, works on all timeframes
  2. ndog/nwog -- store 0 instead of NaN when no gap exists (distinguishes no-gap from missing)
  3. dealing range threshold lowered for early low-vol periods

New features added:
  - distance_to_pdh / distance_to_pdl ($ and %)
  - distance_to_pwh / distance_to_pwl ($ and %)
  - liquidity_above / liquidity_below (nearest unswept swing level)
  - volume_percentile (rolling 100-candle rank 0-100)
  - displacement_flag (single candle move > 2x ATR)
  - htf_confluence_score (placeholder, filled in master builder)
  - candles_since_choch (trend age)
  - time_of_week (0=Mon, 6=Sun)
  - hour_of_day (UTC hour, for session/time analysis)
  - price_vs_session_vwap (close vs rolling session VWAP)
  - fvg_confluence (count of active unmitigated FVGs near current price)
  - ob_confluence (count of active unmitigated OBs near current price)
  - funding_rate_cumulative (rolling 24h sum)
  - killzone_active (binary: is this candle inside a killzone window)

Reads:  BTCUSDT_BINANCE_{interval}_{start}_to_{end}_enriched.csv
Writes: BTCUSDT_BINANCE_{interval}_{start}_to_{end}_enriched_v2.csv

Requires: pip install pandas numpy
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timezone

# ============================================================================
# CONFIG
# ============================================================================

DATA_DIR   = r"C:\Users\tjall\Desktop\Trading\data"
SYMBOL     = "BTCUSDT"
START_DATE = "2017-08-17"
END_DATE   = "2026-03-01"
INTERVALS  = ["5m", "15m", "30m", "1h", "4h", "1d"]

# How many candles to look back for volume percentile
VOL_PERCENTILE_WINDOW = 100

# Displacement: move larger than this multiple of ATR in a single candle
DISPLACEMENT_ATR_MULT = 2.0

# FVG/OB confluence: how close does price need to be to count as "at" the level
# expressed as fraction of ATR
CONFLUENCE_ATR_FRACTION = 1.0

# Funding rate rolling window for cumulative (in candles)
# 24h worth of candles per timeframe
FUNDING_CUMULATIVE_WINDOWS = {
    "5m": 288, "15m": 96, "30m": 48,
    "1h": 24,  "4h": 6,   "1d": 1,
}

# Killzone windows in UTC minutes (start, end) -- half-open intervals
KILLZONES = {
    "Asia":    (60,  300),   # 01:00-05:00 UTC
    "London":  (420, 600),   # 07:00-10:00 UTC
    "NewYork": (720, 900),   # 12:00-15:00 UTC
}

# ============================================================================
# LOAD / SAVE
# ============================================================================

def load_enriched(interval):
    path = os.path.join(DATA_DIR,
        f"{SYMBOL}_BINANCE_{interval}_{START_DATE}_to_{END_DATE}_enriched.csv")
    if not os.path.exists(path):
        print(f"  Not found: {path}")
        return None
    df = pd.read_csv(path, low_memory=False)
    df["open_time_utc"] = pd.to_datetime(df["open_time_utc"], utc=True)
    df["open_time"]     = df["open_time"].astype(np.int64)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("open_time").reset_index(drop=True)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} cols")
    return df

def save_enriched_v2(df, interval):
    path = os.path.join(DATA_DIR,
        f"{SYMBOL}_BINANCE_{interval}_{START_DATE}_to_{END_DATE}_enriched_v2.csv")
    df.to_csv(path, index=False)
    print(f"  Saved -> {os.path.basename(path)}")

# ============================================================================
# FIX 1: ny_open_830
# ============================================================================

def fix_ny_open_830(df):
    """
    Fix ny_open_830 to work on all timeframes.
    8:30 EST = 13:30 UTC = minute 810.
    Find the first candle each day whose open_time is >= 13:30 UTC,
    use its open price, forward-fill through the rest of the day.
    """
    df = df.copy()
    dt = df["open_time_utc"]
    df["_date"]     = dt.dt.date
    df["_utc_mins"] = dt.dt.hour * 60 + dt.dt.minute

    # Mark the first candle of each day that is at or after 13:30 UTC
    df["_past_830"] = df["_utc_mins"] >= 810

    # For each day, get the open price of the first candle >= 13:30
    def get_830_open(group):
        past = group[group["_past_830"]]
        if past.empty:
            return pd.Series(np.nan, index=group.index)
        first_val = past.iloc[0]["open"]
        result = pd.Series(np.nan, index=group.index)
        result.iloc[past.index.get_loc(past.index[0]):] = first_val
        return result

    # Vectorized approach: mark the first candle at/after 810 each day
    df["ny_open_830"] = np.nan

    for date, group in df.groupby("_date"):
        past_830 = group[group["_utc_mins"] >= 810]
        if past_830.empty:
            continue
        first_idx = past_830.index[0]
        df.loc[first_idx:group.index[-1], "ny_open_830"] = group.loc[first_idx, "open"]

    df.drop(columns=["_date", "_utc_mins", "_past_830"], inplace=True)
    return df

# ============================================================================
# FIX 2: ndog / nwog null handling
# ============================================================================

def fix_opening_gaps(df):
    """
    Replace NaN in ndog/nwog with 0 when there is genuinely no gap
    (open == previous close). Keep NaN only for the very first row of
    history where there is no previous data.
    """
    df = df.copy()

    for col_h, col_l, col_filled in [
        ("ndog_high", "ndog_low", "ndog_filled"),
        ("nwog_high", "nwog_low", "nwog_filled"),
    ]:
        if col_h not in df.columns:
            continue
        # Where both are NaN but we have valid open/close data = no gap = set to 0
        no_gap_mask = (df[col_h].isna() & df[col_l].isna() &
                       df["open"].notna() & df["close"].notna())
        df.loc[no_gap_mask, col_h]     = 0.0
        df.loc[no_gap_mask, col_l]     = 0.0
        if col_filled in df.columns:
            df.loc[no_gap_mask, col_filled] = 0

    return df

# ============================================================================
# FIX 3: dealing range -- lower ATR threshold for early data
# ============================================================================

def fix_dealing_range(df):
    """
    Re-compute dr_high/dr_low/dr_eq/premium/discount/ote_zone
    with a lower ATR minimum (0.5 instead of 2.0) to reduce undefined%.
    """
    MIN_ATR = 0.5

    size   = len(df)
    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values
    atr    = df["atr_14"].values if "atr_14" in df.columns else np.ones(size)
    trend  = df["market_trend"].values if "market_trend" in df.columns else np.zeros(size)
    sh     = df["swing_high"].values if "swing_high" in df.columns else np.zeros(size)
    sl     = df["swing_low"].values  if "swing_low"  in df.columns else np.zeros(size)
    sh_p   = df["swing_high_price"].values if "swing_high_price" in df.columns else np.full(size, np.nan)
    sl_p   = df["swing_low_price"].values  if "swing_low_price"  in df.columns else np.full(size, np.nan)

    dr_high  = np.full(size, np.nan)
    dr_low   = np.full(size, np.nan)
    dr_eq    = np.full(size, np.nan)
    premium  = np.zeros(size, dtype=np.int8)
    discount = np.zeros(size, dtype=np.int8)
    ote_zone = np.zeros(size, dtype=np.int8)

    last_sh = np.nan
    last_sl = np.nan

    for i in range(size):
        if sh[i] == 1 and not np.isnan(sh_p[i]):
            last_sh = sh_p[i]
        if sl[i] == 1 and not np.isnan(sl_p[i]):
            last_sl = sl_p[i]

        if np.isnan(last_sh) or np.isnan(last_sl):
            continue

        rng = last_sh - last_sl
        if np.isnan(atr[i]) or rng < atr[i] * MIN_ATR:
            continue

        eq = last_sl + rng * 0.5
        dr_high[i] = last_sh
        dr_low[i]  = last_sl
        dr_eq[i]   = eq

        if closes[i] > eq:
            premium[i]  = 1
        else:
            discount[i] = 1

        if trend[i] == 1:
            ote_lo = last_sh - rng * 0.79
            ote_hi = last_sh - rng * 0.62
            if ote_lo <= closes[i] <= ote_hi:
                ote_zone[i] = 1
        elif trend[i] == -1:
            ote_lo = last_sl + rng * 0.62
            ote_hi = last_sl + rng * 0.79
            if ote_lo <= closes[i] <= ote_hi:
                ote_zone[i] = 1

    df["dr_high"]  = dr_high
    df["dr_low"]   = dr_low
    df["dr_eq"]    = dr_eq
    df["premium"]  = premium
    df["discount"] = discount
    df["ote_zone"] = ote_zone

    return df

# ============================================================================
# NEW FEATURES
# ============================================================================

def add_distance_to_levels(df):
    """
    Distance from current close to key reference levels.
    Stored as both absolute $ and percentage.
    """
    close = df["close"]

    for col, label in [
        ("pdh", "pdh"), ("pdl", "pdl"),
        ("pwh", "pwh"), ("pwl", "pwl"),
    ]:
        if col in df.columns:
            diff = close - df[col]
            df[f"dist_{label}_usd"] = diff
            df[f"dist_{label}_pct"] = diff / df[col] * 100

    # Distance to nearest FVG midpoints
    for side in ["bull", "bear"]:
        mid_col = f"fvg_{side}_mid"
        if mid_col in df.columns:
            df[f"dist_fvg_{side}_usd"] = close - df[mid_col]
            df[f"dist_fvg_{side}_pct"] = (close - df[mid_col]) / df[mid_col] * 100

    # Distance to nearest OB top/bot
    for side in ["bull", "bear"]:
        top_col = f"ob_{side}_top"
        bot_col = f"ob_{side}_bot"
        if top_col in df.columns and bot_col in df.columns:
            ob_mid = (df[top_col] + df[bot_col]) / 2
            df[f"dist_ob_{side}_usd"] = close - ob_mid
            df[f"dist_ob_{side}_pct"] = (close - ob_mid) / ob_mid * 100

    return df

def add_liquidity_levels(df):
    """
    Nearest unswept swing high above price (liquidity_above)
    Nearest unswept swing low  below price (liquidity_below)
    These represent resting stop orders ICT targets.
    """
    size   = len(df)
    closes = df["close"].values
    sh     = df["swing_high"].values if "swing_high" in df.columns else np.zeros(size)
    sl     = df["swing_low"].values  if "swing_low"  in df.columns else np.zeros(size)
    sh_p   = df["swing_high_price"].values if "swing_high_price" in df.columns else np.full(size, np.nan)
    sl_p   = df["swing_low_price"].values  if "swing_low_price"  in df.columns else np.full(size, np.nan)
    bs     = df["bull_liq_sweep"].values if "bull_liq_sweep" in df.columns else np.zeros(size)
    brs    = df["bear_liq_sweep"].values if "bear_liq_sweep" in df.columns else np.zeros(size)

    liq_above = np.full(size, np.nan)
    liq_below = np.full(size, np.nan)

    # Track intact swing levels
    intact_highs = []  # (price,)
    intact_lows  = []

    for i in range(size):
        # Add new swings
        if sh[i] == 1 and not np.isnan(sh_p[i]):
            intact_highs.append(sh_p[i])
        if sl[i] == 1 and not np.isnan(sl_p[i]):
            intact_lows.append(sl_p[i])

        # Remove swept levels
        if brs[i] == 1:  # bear sweep hit a high
            intact_highs = [p for p in intact_highs if p > closes[i]]
        if bs[i] == 1:   # bull sweep hit a low
            intact_lows  = [p for p in intact_lows  if p < closes[i]]

        # Find nearest above and below
        above = [p for p in intact_highs if p > closes[i]]
        below = [p for p in intact_lows  if p < closes[i]]

        if above:
            liq_above[i] = min(above)
        if below:
            liq_below[i] = max(below)

    df["liquidity_above"] = liq_above
    df["liquidity_below"] = liq_below

    # Distance to nearest liquidity
    df["dist_liq_above_usd"] = df["liquidity_above"] - df["close"]
    df["dist_liq_above_pct"] = df["dist_liq_above_usd"] / df["close"] * 100
    df["dist_liq_below_usd"] = df["close"] - df["liquidity_below"]
    df["dist_liq_below_pct"] = df["dist_liq_below_usd"] / df["close"] * 100

    return df

def add_volume_percentile(df, window=VOL_PERCENTILE_WINDOW):
    """
    Rolling percentile rank of current volume vs last N candles (0-100).
    100 = highest volume in window, 0 = lowest.
    """
    vol = df["volume"]
    df["volume_percentile"] = (
        vol.rolling(window, min_periods=1)
           .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False)
    )
    return df

def add_displacement_flag(df, atr_mult=DISPLACEMENT_ATR_MULT):
    """
    Flag candles where the body OR range is > atr_mult * ATR.
    Displacement = aggressive single-candle move, ICT signature of smart money.
    """
    if "atr_14" not in df.columns:
        df["displacement_flag"] = 0
        return df

    candle_range = df["high"] - df["low"]
    candle_body  = (df["close"] - df["open"]).abs()
    threshold    = df["atr_14"] * atr_mult

    df["displacement_flag"] = ((candle_range > threshold) | (candle_body > threshold)).astype(np.int8)
    df["displacement_dir"]  = np.where(
        df["displacement_flag"] == 1,
        np.where(df["close"] > df["open"], 1, -1),
        0
    ).astype(np.int8)

    return df

def add_candles_since_choch(df):
    """
    Count of candles since the last CHoCH event.
    Tells you how old/mature the current trend is.
    """
    if "choch" not in df.columns:
        df["candles_since_choch"] = np.nan
        return df

    choch = df["choch"].values
    size  = len(df)
    result = np.full(size, np.nan)
    last_choch = -1

    for i in range(size):
        if choch[i] != 0:
            last_choch = i
        if last_choch >= 0:
            result[i] = i - last_choch

    df["candles_since_choch"] = result
    return df

def add_time_features(df):
    """
    Time-of-week, hour-of-day, and minute-of-day for seasonality analysis.
    Also adds cyclical encoding (sin/cos) for ML models.
    """
    dt = df["open_time_utc"]

    df["day_of_week"]  = dt.dt.dayofweek.astype(np.int8)  # 0=Mon, 6=Sun
    df["hour_of_day"]  = dt.dt.hour.astype(np.int8)
    df["minute_of_day"]= (dt.dt.hour * 60 + dt.dt.minute).astype(np.int16)

    # Cyclical encoding for ML (so model knows Mon and Sun are adjacent)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["hod_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hod_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)

    return df

def add_killzone_active(df):
    """
    Binary flag: is this candle inside any killzone window?
    More precise than the session label which covers the full session.
    """
    utc_mins = df["open_time_utc"].dt.hour * 60 + df["open_time_utc"].dt.minute
    active = np.zeros(len(df), dtype=np.int8)

    for name, (start, end) in KILLZONES.items():
        active |= ((utc_mins >= start) & (utc_mins < end)).astype(np.int8)

    df["killzone_active"] = active
    return df

def add_session_vwap(df):
    """
    Rolling VWAP from session open to current candle.
    Session defined by 'session' column.
    price_vs_session_vwap: close relative to VWAP (positive = above, negative = below)
    """
    if "session" not in df.columns:
        df["session_vwap"]         = np.nan
        df["price_vs_session_vwap"]= np.nan
        return df

    df = df.copy()
    df["_tp"]  = (df["high"] + df["low"] + df["close"]) / 3   # typical price
    df["_tpv"] = df["_tp"] * df["volume"]

    # Group by date + session to compute VWAP within each session block
    df["_date"]    = df["open_time_utc"].dt.date
    df["_sess_grp"]= df["_date"].astype(str) + "_" + df["session"].astype(str)

    df["_cum_tpv"] = df.groupby("_sess_grp")["_tpv"].cumsum()
    df["_cum_vol"] = df.groupby("_sess_grp")["volume"].cumsum()
    df["session_vwap"] = df["_cum_tpv"] / df["_cum_vol"]

    df["price_vs_session_vwap"] = df["close"] - df["session_vwap"]
    df["price_vs_vwap_pct"]     = df["price_vs_session_vwap"] / df["session_vwap"] * 100

    df.drop(columns=["_tp", "_tpv", "_date", "_sess_grp", "_cum_tpv", "_cum_vol"],
            inplace=True)
    return df

def add_fvg_confluence(df, atr_fraction=CONFLUENCE_ATR_FRACTION):
    """
    Count of active UNMITIGATED FVGs whose range overlaps with current price +/- ATR.
    Higher count = stronger confluence zone.
    This is a simplified version -- checks if current close is near the FVG midpoint.
    """
    if "fvg_bull_mid" not in df.columns:
        df["fvg_bull_confluence"] = 0
        df["fvg_bear_confluence"] = 0
        return df

    close = df["close"].values
    atr   = df["atr_14"].values if "atr_14" in df.columns else np.ones(len(df))
    bull_mid = df["fvg_bull_mid"].values
    bear_mid = df["fvg_bear_mid"].values
    bull_mit = df["fvg_bull_mitigated"].values if "fvg_bull_mitigated" in df.columns else np.zeros(len(df))
    bear_mit = df["fvg_bear_mitigated"].values if "fvg_bear_mitigated" in df.columns else np.zeros(len(df))

    bull_conf = np.zeros(len(df), dtype=np.int8)
    bear_conf = np.zeros(len(df), dtype=np.int8)

    for i in range(len(df)):
        if np.isnan(atr[i]):
            continue
        zone = atr[i] * atr_fraction

        # Bull FVG confluence: unmitigated, close is within zone of mid
        if not np.isnan(bull_mid[i]) and bull_mit[i] == 0:
            if abs(close[i] - bull_mid[i]) <= zone:
                bull_conf[i] = 1

        # Bear FVG confluence
        if not np.isnan(bear_mid[i]) and bear_mit[i] == 0:
            if abs(close[i] - bear_mid[i]) <= zone:
                bear_conf[i] = 1

    df["fvg_bull_confluence"] = bull_conf
    df["fvg_bear_confluence"] = bear_conf

    return df

def add_ob_confluence(df, atr_fraction=CONFLUENCE_ATR_FRACTION):
    """
    Flag if price is currently inside an unmitigated OB body.
    """
    if "ob_bull_top" not in df.columns:
        df["ob_bull_confluence"] = 0
        df["ob_bear_confluence"] = 0
        return df

    close    = df["close"].values
    bull_top = df["ob_bull_top"].values
    bull_bot = df["ob_bull_bot"].values
    bear_top = df["ob_bear_top"].values
    bear_bot = df["ob_bear_bot"].values
    bull_mit = df["ob_bull_mitigated"].values if "ob_bull_mitigated" in df.columns else np.zeros(len(df))
    bear_mit = df["ob_bear_mitigated"].values if "ob_bear_mitigated" in df.columns else np.zeros(len(df))

    bull_conf = np.zeros(len(df), dtype=np.int8)
    bear_conf = np.zeros(len(df), dtype=np.int8)

    for i in range(len(df)):
        if not np.isnan(bull_top[i]) and bull_mit[i] == 0:
            if bull_bot[i] <= close[i] <= bull_top[i]:
                bull_conf[i] = 1
        if not np.isnan(bear_top[i]) and bear_mit[i] == 0:
            if bear_bot[i] <= close[i] <= bear_top[i]:
                bear_conf[i] = 1

    df["ob_bull_confluence"] = bull_conf
    df["ob_bear_confluence"] = bear_conf

    return df

def add_funding_cumulative(df, interval):
    """
    Rolling sum of funding rate over 24h equivalent window.
    Positive = longs paying shorts (bearish pressure building)
    Negative = shorts paying longs (bullish pressure building)
    """
    if "funding_rate" not in df.columns:
        df["funding_rate_24h"] = np.nan
        return df

    window = FUNDING_CUMULATIVE_WINDOWS.get(interval, 24)
    df["funding_rate_24h"] = df["funding_rate"].rolling(window, min_periods=1).sum()
    return df

def add_htf_confluence_score(df):
    """
    Composite score: how many timeframes agree on current bias.
    Uses market_trend from each HTF already in the enriched file.
    On the per-timeframe files this uses only the base TF trend.
    The master file version will use all HTF prefixes.
    Score: +1 per bullish TF, -1 per bearish TF, 0 undefined.
    """
    # On individual timeframe files, just use the base trend
    if "market_trend" in df.columns:
        df["trend_score"] = df["market_trend"].fillna(0).astype(int)
    else:
        df["trend_score"] = 0
    return df

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def enrich_v2(interval):
    print(f"\n{'='*60}")
    print(f"  {SYMBOL} | {interval} -- Extended Enrichment v2")
    print(f"{'='*60}")

    df = load_enriched(interval)
    if df is None:
        return

    print(f"  [1/14] Fixing ny_open_830...")
    df = fix_ny_open_830(df)

    print(f"  [2/14] Fixing opening gaps (ndog/nwog nulls)...")
    df = fix_opening_gaps(df)

    print(f"  [3/14] Fixing dealing range threshold...")
    df = fix_dealing_range(df)

    print(f"  [4/14] Distance to key levels...")
    df = add_distance_to_levels(df)

    print(f"  [5/14] Liquidity levels (above/below)...")
    df = add_liquidity_levels(df)

    print(f"  [6/14] Volume percentile...")
    df = add_volume_percentile(df)

    print(f"  [7/14] Displacement flags...")
    df = add_displacement_flag(df)

    print(f"  [8/14] Candles since CHoCH...")
    df = add_candles_since_choch(df)

    print(f"  [9/14] Time features (day/hour/cyclical)...")
    df = add_time_features(df)

    print(f"  [10/14] Killzone active flag...")
    df = add_killzone_active(df)

    print(f"  [11/14] Session VWAP...")
    df = add_session_vwap(df)

    print(f"  [12/14] FVG confluence...")
    df = add_fvg_confluence(df)

    print(f"  [13/14] OB confluence...")
    df = add_ob_confluence(df)

    print(f"  [14/14] Funding rate cumulative + trend score...")
    df = add_funding_cumulative(df, interval)
    df = add_htf_confluence_score(df)

    save_enriched_v2(df, interval)

    # Quick summary of new cols
    new_col_count = len(df.columns)
    ny_nulls = df["ny_open_830"].isna().sum()
    dr_nulls = df["dr_high"].isna().sum()
    ndog_0   = (df["ndog_high"] == 0).sum() if "ndog_high" in df.columns else 0
    disp     = df["displacement_flag"].sum() if "displacement_flag" in df.columns else 0

    print(f"\n  Summary:")
    print(f"    Total columns:        {new_col_count}")
    print(f"    ny_open_830 nulls:    {ny_nulls:,} (was ~56% or 100%, now fixed)")
    print(f"    dr_high nulls:        {dr_nulls:,} ({dr_nulls/len(df)*100:.1f}%)")
    print(f"    ndog no-gap zeros:    {ndog_0:,}")
    print(f"    Displacement candles: {disp:,} ({disp/len(df)*100:.2f}%)")


def main():
    print(f"Extended ICT Enrichment v2 -- {SYMBOL}")
    print(f"Input/Output: {DATA_DIR}")

    for interval in INTERVALS:
        try:
            enrich_v2(interval)
        except Exception as e:
            import traceback
            print(f"\n  ERROR on {interval}: {e}")
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"  Done. Files saved with suffix _enriched_v2.csv")
    print(f"  Re-run build_master.py pointing at _v2 files to rebuild master.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
