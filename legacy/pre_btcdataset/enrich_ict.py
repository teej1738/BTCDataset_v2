"""
Script 2 — enrich_ict.py
ICT Feature Enrichment for BTCUSDT Binance Futures Data

Reads raw CSVs produced by download_btcusdt.py and adds:
  - Fair Value Gaps (bullish + bearish, mitigation at 50%)
  - Order Blocks (body only, bullish + bearish, ATR-based significance)
  - Swing Highs / Swing Lows (timeframe-aware lookback)
  - Liquidity Sweeps (wick through swing H/L, closes back inside)
  - Break of Structure (BOS) / Change of Character (CHoCH)
  - Market structure trend label (Bullish / Bearish / Ranging)
  - Premium / Discount zone relative to current dealing range
  - OTE zone (62-79% retracement)
  - Dealing range high/low stored on every candle
  - NDOG / NWOG (new day/week opening gaps)

Requires: pip install pandas numpy
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timezone
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — tune these without touching any logic below
# ══════════════════════════════════════════════════════════════════════════════

INPUT_DIR  = r"C:\Users\tjall\Desktop\Trading\data"
OUTPUT_DIR = r"C:\Users\tjall\Desktop\Trading\data"

SYMBOL     = "BTCUSDT"
START_DATE = "2017-08-17"
END_DATE   = "2026-03-01"

# Swing lookback — how many candles each side to confirm a swing high/low
# Timeframe-aware: larger TF = fewer candles needed to confirm significance
SWING_LOOKBACK = {
    "5m":  10,
    "15m": 8,
    "30m": 6,
    "1h":  5,
    "4h":  4,
    "1d":  3,
}

# Order Block: minimum move away from OB (in ATR multiples) to be considered significant
OB_ATR_MULTIPLIER = 1.5

# FVG: minimum gap size as fraction of ATR to filter noise
FVG_MIN_ATR_FRACTION = 0.1

# OTE retracement zone (Fibonacci)
OTE_LOW  = 0.62
OTE_HIGH = 0.79

# Dealing range: minimum swing size (ATR multiples) to qualify as a dealing range boundary
DEALING_RANGE_MIN_ATR = 2.0

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_csv(interval):
    filename = f"{SYMBOL}_BINANCE_{interval}_{START_DATE}_to_{END_DATE}.csv"
    path = os.path.join(INPUT_DIR, filename)
    if not os.path.exists(path):
        print(f"  File not found: {path}")
        return None
    df = pd.read_csv(path, low_memory=False)
    df["open_time_utc"] = pd.to_datetime(df["open_time_utc"], utc=True)
    df["open_time"]     = df["open_time"].astype(np.int64)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df = df.sort_values("open_time").reset_index(drop=True)
    print(f"  Loaded {len(df):,} rows for {interval}")
    return df

def save_csv(df, interval):
    filename = f"{SYMBOL}_BINANCE_{interval}_{START_DATE}_to_{END_DATE}_enriched.csv"
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, index=False)
    print(f"  Saved {len(df):,} rows -> {filename}")

# ══════════════════════════════════════════════════════════════════════════════
# SWING HIGHS / LOWS
# ══════════════════════════════════════════════════════════════════════════════

def add_swings(df, lookback):
    """
    Detect swing highs and lows using a rolling window.
    A swing high at index i means high[i] is the highest in the window [i-n : i+n].
    A swing low  at index i means low[i]  is the lowest  in the window [i-n : i+n].
    Flags: 1 = confirmed, 0 = not a swing point.
    Also stores the price level of each swing for later use.
    """
    n = lookback
    highs = df["high"].values
    lows  = df["low"].values
    size  = len(df)

    swing_high_flag  = np.zeros(size, dtype=np.int8)
    swing_low_flag   = np.zeros(size, dtype=np.int8)
    swing_high_price = np.full(size, np.nan)
    swing_low_price  = np.full(size, np.nan)

    for i in range(n, size - n):
        window_h = highs[i-n : i+n+1]
        window_l = lows[i-n  : i+n+1]
        if highs[i] == np.max(window_h):
            swing_high_flag[i]  = 1
            swing_high_price[i] = highs[i]
        if lows[i] == np.min(window_l):
            swing_low_flag[i]  = 1
            swing_low_price[i] = lows[i]

    df["swing_high"]       = swing_high_flag
    df["swing_high_price"] = swing_high_price
    df["swing_low"]        = swing_low_flag
    df["swing_low_price"]  = swing_low_price

    return df

# ══════════════════════════════════════════════════════════════════════════════
# LIQUIDITY SWEEPS
# ══════════════════════════════════════════════════════════════════════════════

def add_liquidity_sweeps(df):
    """
    A bullish liquidity sweep: candle wicks BELOW a prior swing low but CLOSES above it.
    A bearish liquidity sweep: candle wicks ABOVE a prior swing high but CLOSES below it.
    Tracks the most recent unswept swing high/low at each point in time.
    """
    size   = len(df)
    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values
    sh     = df["swing_high_flag"].values if "swing_high_flag" in df.columns else df["swing_high"].values
    sl     = df["swing_low_flag"].values  if "swing_low_flag"  in df.columns else df["swing_low"].values
    sh_p   = df["swing_high_price"].values
    sl_p   = df["swing_low_price"].values

    bull_sweep = np.zeros(size, dtype=np.int8)  # swept below prior swing low and closed above
    bear_sweep = np.zeros(size, dtype=np.int8)  # swept above prior swing high and closed below
    sweep_level = np.full(size, np.nan)          # price level that was swept

    # Rolling list of intact swing highs/lows
    intact_highs = []  # list of (index, price)
    intact_lows  = []  # list of (index, price)

    for i in range(size):
        # Register new swing points
        if sh[i] == 1 and not np.isnan(sh_p[i]):
            intact_highs.append((i, sh_p[i]))
        if sl[i] == 1 and not np.isnan(sl_p[i]):
            intact_lows.append((i, sl_p[i]))

        # Check for bearish sweep (wick above recent swing high, close below it)
        swept_highs = []
        for idx, price in intact_highs:
            if highs[i] > price and closes[i] < price:
                bear_sweep[i]  = 1
                sweep_level[i] = price
                swept_highs.append((idx, price))
        intact_highs = [(idx, p) for idx, p in intact_highs if (idx, p) not in swept_highs]

        # Check for bullish sweep (wick below recent swing low, close above it)
        swept_lows = []
        for idx, price in intact_lows:
            if lows[i] < price and closes[i] > price:
                bull_sweep[i]  = 1
                sweep_level[i] = price
                swept_lows.append((idx, price))
        intact_lows = [(idx, p) for idx, p in intact_lows if (idx, p) not in swept_lows]

    df["bull_liq_sweep"]  = bull_sweep
    df["bear_liq_sweep"]  = bear_sweep
    df["sweep_level"]     = sweep_level

    return df

# ══════════════════════════════════════════════════════════════════════════════
# MARKET STRUCTURE — BOS / CHoCH
# ══════════════════════════════════════════════════════════════════════════════

def add_market_structure(df):
    """
    Track market structure using swing highs/lows.
    
    Rules:
    - Bullish BOS:  price closes above a prior swing high while in uptrend -> continuation
    - Bearish BOS:  price closes below a prior swing low  while in downtrend -> continuation
    - Bullish CHoCH: price closes above a prior swing high while in downtrend -> trend reversal
    - Bearish CHoCH: price closes below a prior swing low  while in uptrend  -> trend reversal
    
    trend: 1 = bullish, -1 = bearish, 0 = undefined (start)
    """
    size   = len(df)
    closes = df["close"].values
    sh     = df["swing_high"].values
    sl     = df["swing_low"].values
    sh_p   = df["swing_high_price"].values
    sl_p   = df["swing_low_price"].values

    bos   = np.zeros(size, dtype=np.int8)   # 1=bull BOS, -1=bear BOS
    choch = np.zeros(size, dtype=np.int8)   # 1=bull CHoCH, -1=bear CHoCH
    trend = np.zeros(size, dtype=np.int8)   # 1=bull, -1=bear, 0=undefined

    last_sh_price = np.nan
    last_sl_price = np.nan
    current_trend = 0

    for i in range(size):
        # Update last known swing levels
        if sh[i] == 1 and not np.isnan(sh_p[i]):
            last_sh_price = sh_p[i]
        if sl[i] == 1 and not np.isnan(sl_p[i]):
            last_sl_price = sl_p[i]

        # Check for structure breaks
        if not np.isnan(last_sh_price) and closes[i] > last_sh_price:
            if current_trend == 1:
                bos[i] = 1        # bullish continuation
            elif current_trend == -1:
                choch[i] = 1      # bullish reversal
            current_trend = 1

        elif not np.isnan(last_sl_price) and closes[i] < last_sl_price:
            if current_trend == -1:
                bos[i] = -1       # bearish continuation
            elif current_trend == 1:
                choch[i] = -1     # bearish reversal
            current_trend = -1

        trend[i] = current_trend

    df["bos"]            = bos
    df["choch"]          = choch
    df["market_trend"]   = trend  # 1=bullish, -1=bearish, 0=undefined

    return df

# ══════════════════════════════════════════════════════════════════════════════
# FAIR VALUE GAPS
# ══════════════════════════════════════════════════════════════════════════════

def add_fvg(df, min_atr_fraction=FVG_MIN_ATR_FRACTION):
    """
    Bullish FVG: candle[i-1].high < candle[i+1].low  (gap up, price left unfilled space)
    Bearish FVG: candle[i-1].low  > candle[i+1].high (gap down)
    
    Stored columns per candle:
      fvg_bull        — 1 if this candle IS the middle candle of a bullish FVG
      fvg_bear        — 1 if this candle IS the middle candle of a bearish FVG
      fvg_bull_top    — top of the most recent active bullish FVG
      fvg_bull_bot    — bottom of the most recent active bullish FVG
      fvg_bull_mid    — 50% midpoint (mitigation target)
      fvg_bull_mitigated — 1 if the most recent bull FVG has been mitigated
      fvg_bear_top / bot / mid / mitigated — same for bearish
      fvg_bull_age    — candles since the most recent bull FVG formed
      fvg_bear_age    — candles since the most recent bear FVG formed
    """
    size  = len(df)
    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values
    atr    = df[f"atr_14"].values if "atr_14" in df.columns else np.ones(size)

    # Detect FVG formation candles
    bull_fvg_flag = np.zeros(size, dtype=np.int8)
    bear_fvg_flag = np.zeros(size, dtype=np.int8)
    bull_fvg_top  = np.full(size, np.nan)
    bull_fvg_bot  = np.full(size, np.nan)
    bear_fvg_top  = np.full(size, np.nan)
    bear_fvg_bot  = np.full(size, np.nan)

    for i in range(1, size - 1):
        gap_size_bull = lows[i+1]  - highs[i-1]
        gap_size_bear = lows[i-1]  - highs[i+1]
        min_size      = atr[i] * min_atr_fraction if not np.isnan(atr[i]) else 0

        if gap_size_bull > min_size:
            bull_fvg_flag[i] = 1
            bull_fvg_top[i]  = lows[i+1]
            bull_fvg_bot[i]  = highs[i-1]

        if gap_size_bear > min_size:
            bear_fvg_flag[i] = 1
            bear_fvg_top[i]  = lows[i-1]
            bear_fvg_bot[i]  = highs[i+1]

    df["fvg_bull"] = bull_fvg_flag
    df["fvg_bear"] = bear_fvg_flag

    # Now forward-fill the most recent active FVG state onto every candle
    # Track mitigation (price reaches 50% of gap)
    curr_bull_top = curr_bull_bot = curr_bull_mid = np.nan
    curr_bear_top = curr_bear_bot = curr_bear_mid = np.nan
    bull_mitigated = bear_mitigated = 0
    bull_age = bear_age = 0
    bull_formed = bear_formed = False

    out_bull_top  = np.full(size, np.nan)
    out_bull_bot  = np.full(size, np.nan)
    out_bull_mid  = np.full(size, np.nan)
    out_bull_mit  = np.zeros(size, dtype=np.int8)
    out_bull_age  = np.full(size, np.nan)
    out_bear_top  = np.full(size, np.nan)
    out_bear_bot  = np.full(size, np.nan)
    out_bear_mid  = np.full(size, np.nan)
    out_bear_mit  = np.zeros(size, dtype=np.int8)
    out_bear_age  = np.full(size, np.nan)

    for i in range(size):
        # New FVG formed
        if bull_fvg_flag[i] == 1:
            curr_bull_top  = bull_fvg_top[i]
            curr_bull_bot  = bull_fvg_bot[i]
            curr_bull_mid  = (curr_bull_top + curr_bull_bot) / 2
            bull_mitigated = 0
            bull_age       = 0
            bull_formed    = True

        if bear_fvg_flag[i] == 1:
            curr_bear_top  = bear_fvg_top[i]
            curr_bear_bot  = bear_fvg_bot[i]
            curr_bear_mid  = (curr_bear_top + curr_bear_bot) / 2
            bear_mitigated = 0
            bear_age       = 0
            bear_formed    = True

        # Check mitigation (price reaches 50% midpoint of gap)
        if bull_formed and not bull_mitigated and not np.isnan(curr_bull_mid):
            if lows[i] <= curr_bull_mid:
                bull_mitigated = 1

        if bear_formed and not bear_mitigated and not np.isnan(curr_bear_mid):
            if highs[i] >= curr_bear_mid:
                bear_mitigated = 1

        # Write state
        if bull_formed:
            out_bull_top[i] = curr_bull_top
            out_bull_bot[i] = curr_bull_bot
            out_bull_mid[i] = curr_bull_mid
            out_bull_mit[i] = bull_mitigated
            out_bull_age[i] = bull_age
            bull_age += 1

        if bear_formed:
            out_bear_top[i] = curr_bear_top
            out_bear_bot[i] = curr_bear_bot
            out_bear_mid[i] = curr_bear_mid
            out_bear_mit[i] = bear_mitigated
            out_bear_age[i] = bear_age
            bear_age += 1

    df["fvg_bull_top"]        = out_bull_top
    df["fvg_bull_bot"]        = out_bull_bot
    df["fvg_bull_mid"]        = out_bull_mid
    df["fvg_bull_mitigated"]  = out_bull_mit
    df["fvg_bull_age"]        = out_bull_age
    df["fvg_bear_top"]        = out_bear_top
    df["fvg_bear_bot"]        = out_bear_bot
    df["fvg_bear_mid"]        = out_bear_mid
    df["fvg_bear_mitigated"]  = out_bear_mit
    df["fvg_bear_age"]        = out_bear_age

    return df

# ══════════════════════════════════════════════════════════════════════════════
# ORDER BLOCKS
# ══════════════════════════════════════════════════════════════════════════════

def add_order_blocks(df, atr_multiplier=OB_ATR_MULTIPLIER):
    """
    Bullish OB: last BEARISH (red) candle before a significant bullish move up.
               Body = open (top) and close (bottom) of that candle.
    Bearish OB: last BULLISH (green) candle before a significant bearish move down.
               Body = open (bottom) and close (top) of that candle.

    'Significant' = the subsequent move is >= atr_multiplier * ATR.
    Mitigation: price returns into the OB body (trades between OB open and close).
    """
    size   = len(df)
    opens  = df["open"].values
    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values
    atr    = df["atr_14"].values if "atr_14" in df.columns else np.ones(size)

    bull_ob_flag = np.zeros(size, dtype=np.int8)
    bear_ob_flag = np.zeros(size, dtype=np.int8)
    bull_ob_top  = np.full(size, np.nan)   # open of the bearish candle (body top)
    bull_ob_bot  = np.full(size, np.nan)   # close of the bearish candle (body bottom)
    bear_ob_top  = np.full(size, np.nan)   # close of the bullish candle (body top)
    bear_ob_bot  = np.full(size, np.nan)   # open of the bullish candle (body bottom)

    for i in range(1, size - 1):
        if np.isnan(atr[i]):
            continue
        min_move = atr[i] * atr_multiplier

        # Bullish OB: candle[i] is bearish, candle[i+1] moves up significantly
        if closes[i] < opens[i]:   # bearish candle
            subsequent_move = closes[i+1] - closes[i]
            if subsequent_move >= min_move:
                bull_ob_flag[i] = 1
                bull_ob_top[i]  = opens[i]    # body top = open of red candle
                bull_ob_bot[i]  = closes[i]   # body bot = close of red candle

        # Bearish OB: candle[i] is bullish, candle[i+1] moves down significantly
        if closes[i] > opens[i]:   # bullish candle
            subsequent_move = closes[i] - closes[i+1]
            if subsequent_move >= min_move:
                bear_ob_flag[i] = 1
                bear_ob_top[i]  = closes[i]   # body top = close of green candle
                bear_ob_bot[i]  = opens[i]    # body bot = open of green candle

    df["ob_bull"] = bull_ob_flag
    df["ob_bear"] = bear_ob_flag

    # Forward-fill most recent active OB + track mitigation
    curr_bull_top = curr_bull_bot = np.nan
    curr_bear_top = curr_bear_bot = np.nan
    bull_mit = bear_mit = 0
    bull_ob_formed = bear_ob_formed = False
    bull_age = bear_age = 0

    out_bull_top = np.full(size, np.nan)
    out_bull_bot = np.full(size, np.nan)
    out_bull_mit = np.zeros(size, dtype=np.int8)
    out_bull_age = np.full(size, np.nan)
    out_bear_top = np.full(size, np.nan)
    out_bear_bot = np.full(size, np.nan)
    out_bear_mit = np.zeros(size, dtype=np.int8)
    out_bear_age = np.full(size, np.nan)

    for i in range(size):
        if bull_ob_flag[i] == 1:
            curr_bull_top  = bull_ob_top[i]
            curr_bull_bot  = bull_ob_bot[i]
            bull_mit       = 0
            bull_ob_formed = True
            bull_age       = 0

        if bear_ob_flag[i] == 1:
            curr_bear_top  = bear_ob_top[i]
            curr_bear_bot  = bear_ob_bot[i]
            bear_mit       = 0
            bear_ob_formed = True
            bear_age       = 0

        # Mitigation: price trades back into the OB body
        if bull_ob_formed and not bull_mit and not np.isnan(curr_bull_top):
            if lows[i] <= curr_bull_top and highs[i] >= curr_bull_bot:
                bull_mit = 1

        if bear_ob_formed and not bear_mit and not np.isnan(curr_bear_top):
            if highs[i] >= curr_bear_bot and lows[i] <= curr_bear_top:
                bear_mit = 1

        if bull_ob_formed:
            out_bull_top[i] = curr_bull_top
            out_bull_bot[i] = curr_bull_bot
            out_bull_mit[i] = bull_mit
            out_bull_age[i] = bull_age
            bull_age += 1

        if bear_ob_formed:
            out_bear_top[i] = curr_bear_top
            out_bear_bot[i] = curr_bear_bot
            out_bear_mit[i] = bear_mit
            out_bear_age[i] = bear_age
            bear_age += 1

    df["ob_bull_top"]       = out_bull_top
    df["ob_bull_bot"]       = out_bull_bot
    df["ob_bull_mitigated"] = out_bull_mit
    df["ob_bull_age"]       = out_bull_age
    df["ob_bear_top"]       = out_bear_top
    df["ob_bear_bot"]       = out_bear_bot
    df["ob_bear_mitigated"] = out_bear_mit
    df["ob_bear_age"]       = out_bear_age

    return df

# ══════════════════════════════════════════════════════════════════════════════
# DEALING RANGE — PREMIUM / DISCOUNT / OTE
# ══════════════════════════════════════════════════════════════════════════════

def add_dealing_range(df, min_atr=DEALING_RANGE_MIN_ATR):
    """
    The 'dealing range' is the range between the most recent significant
    swing high and swing low (confirmed by market structure).
    
    Premium  = price is above 50% of the range (sell zone in ICT)
    Discount = price is below 50% of the range (buy zone in ICT)
    OTE      = price is in the 62-79% retracement zone
    
    For a bullish market: OTE is 62-79% retracement from swing low to swing high
    For a bearish market: OTE is 62-79% retracement from swing high to swing low
    """
    size   = len(df)
    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values
    atr    = df["atr_14"].values if "atr_14" in df.columns else np.ones(size)
    trend  = df["market_trend"].values if "market_trend" in df.columns else np.zeros(size)
    sh     = df["swing_high"].values
    sl     = df["swing_low"].values
    sh_p   = df["swing_high_price"].values
    sl_p   = df["swing_low_price"].values

    dr_high    = np.full(size, np.nan)
    dr_low     = np.full(size, np.nan)
    dr_eq      = np.full(size, np.nan)   # 50% equilibrium
    premium    = np.zeros(size, dtype=np.int8)
    discount   = np.zeros(size, dtype=np.int8)
    ote_zone   = np.zeros(size, dtype=np.int8)

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
        if np.isnan(atr[i]) or rng < atr[i] * min_atr:
            continue

        eq  = last_sl + rng * 0.5
        dr_high[i] = last_sh
        dr_low[i]  = last_sl
        dr_eq[i]   = eq

        # Premium / Discount
        if closes[i] > eq:
            premium[i]  = 1
        else:
            discount[i] = 1

        # OTE zone — direction-aware
        if trend[i] == 1:
            # Bullish: buying opportunity at 62-79% retracement of the up-swing
            ote_low_price  = last_sh - rng * OTE_HIGH
            ote_high_price = last_sh - rng * OTE_LOW
            if ote_low_price <= closes[i] <= ote_high_price:
                ote_zone[i] = 1
        elif trend[i] == -1:
            # Bearish: selling opportunity at 62-79% retracement of the down-swing
            ote_low_price  = last_sl + rng * OTE_LOW
            ote_high_price = last_sl + rng * OTE_HIGH
            if ote_low_price <= closes[i] <= ote_high_price:
                ote_zone[i] = 1

    df["dr_high"]  = dr_high
    df["dr_low"]   = dr_low
    df["dr_eq"]    = dr_eq
    df["premium"]  = premium
    df["discount"] = discount
    df["ote_zone"] = ote_zone

    return df

# ══════════════════════════════════════════════════════════════════════════════
# NDOG / NWOG
# ══════════════════════════════════════════════════════════════════════════════

def add_opening_gaps(df):
    """
    NDOG (New Day Opening Gap): gap between previous day's close and current day's open.
    NWOG (New Week Opening Gap): gap between previous week's close and current week's open.

    Stored as:
      ndog_high / ndog_low  — top and bottom of the daily gap (NaN if no gap)
      nwog_high / nwog_low  — top and bottom of the weekly gap
      ndog_filled / nwog_filled — 1 once price has traded through the gap
    """
    df = df.copy()
    dt = df["open_time_utc"]

    df["_date"] = dt.dt.date
    df["_week"] = dt.dt.to_period("W")

    # Daily previous close
    daily_close = df.groupby("_date")["close"].last().reset_index()
    daily_close.columns = ["_date", "_prev_day_close"]
    daily_close["_date"] = daily_close["_date"].shift(-1)
    df = df.merge(daily_close, on="_date", how="left")

    # Daily open = first candle of each day
    daily_open = df.groupby("_date")["open"].first().reset_index()
    daily_open.columns = ["_date", "_day_open"]
    df = df.merge(daily_open, on="_date", how="left")

    # NDOG: gap between prev close and today's open
    df["ndog_high"] = np.where(
        df["_day_open"] > df["_prev_day_close"],
        df["_day_open"],
        np.where(df["_day_open"] < df["_prev_day_close"], df["_prev_day_close"], np.nan)
    )
    df["ndog_low"] = np.where(
        df["_day_open"] > df["_prev_day_close"],
        df["_prev_day_close"],
        np.where(df["_day_open"] < df["_prev_day_close"], df["_day_open"], np.nan)
    )

    # Weekly previous close
    weekly_close = df.groupby("_week")["close"].last().reset_index()
    weekly_close.columns = ["_week", "_prev_week_close"]
    weekly_close["_week"] = weekly_close["_week"].shift(-1)
    df = df.merge(weekly_close, on="_week", how="left")

    weekly_open = df.groupby("_week")["open"].first().reset_index()
    weekly_open.columns = ["_week", "_week_open"]
    df = df.merge(weekly_open, on="_week", how="left")

    # NWOG
    df["nwog_high"] = np.where(
        df["_week_open"] > df["_prev_week_close"],
        df["_week_open"],
        np.where(df["_week_open"] < df["_prev_week_close"], df["_prev_week_close"], np.nan)
    )
    df["nwog_low"] = np.where(
        df["_week_open"] > df["_prev_week_close"],
        df["_prev_week_close"],
        np.where(df["_week_open"] < df["_prev_week_close"], df["_week_open"], np.nan)
    )

    # Forward-fill gap levels through the day/week
    df["ndog_high"] = df.groupby("_date")["ndog_high"].transform("ffill")
    df["ndog_low"]  = df.groupby("_date")["ndog_low"].transform("ffill")
    df["nwog_high"] = df.groupby("_week")["nwog_high"].transform("ffill")
    df["nwog_low"]  = df.groupby("_week")["nwog_low"].transform("ffill")

    # Gap filled flag — once price trades through the gap during the day/week
    def mark_filled(group, high_col, low_col):
        filled = np.zeros(len(group), dtype=np.int8)
        h = group["high"].values
        l = group["low"].values
        gh = group[high_col].values
        gl = group[low_col].values
        gap_filled = False
        for i in range(len(group)):
            if gap_filled:
                filled[i] = 1
                continue
            if not np.isnan(gh[i]) and not np.isnan(gl[i]):
                if l[i] <= gh[i] and h[i] >= gl[i]:
                    gap_filled = True
                    filled[i]  = 1
        return filled

    df["ndog_filled"] = np.concatenate([
        mark_filled(g, "ndog_high", "ndog_low")
        for _, g in df.groupby("_date")
    ])
    df["nwog_filled"] = np.concatenate([
        mark_filled(g, "nwog_high", "nwog_low")
        for _, g in df.groupby("_week")
    ])

    df.drop(columns=["_date", "_week", "_prev_day_close", "_day_open",
                     "_prev_week_close", "_week_open"], inplace=True)
    return df

# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENRICHMENT PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

INTERVALS = ["5m", "15m", "30m", "1h", "4h", "1d"]

def enrich(interval):
    print(f"\n{'='*60}")
    print(f"  Enriching {SYMBOL} | {interval}")
    print(f"{'='*60}")

    df = load_csv(interval)
    if df is None:
        return

    lookback = SWING_LOOKBACK.get(interval, 5)

    print(f"  [1/8] Swing highs/lows (lookback={lookback})...")
    df = add_swings(df, lookback)

    print(f"  [2/8] Liquidity sweeps...")
    df = add_liquidity_sweeps(df)

    print(f"  [3/8] Market structure (BOS/CHoCH)...")
    df = add_market_structure(df)

    print(f"  [4/8] Fair Value Gaps...")
    df = add_fvg(df)

    print(f"  [5/8] Order Blocks...")
    df = add_order_blocks(df)

    print(f"  [6/8] Dealing range (Premium/Discount/OTE)...")
    df = add_dealing_range(df)

    print(f"  [7/8] Opening gaps (NDOG/NWOG)...")
    df = add_opening_gaps(df)

    print(f"  [8/8] Saving...")
    save_csv(df, interval)

    # Summary stats
    bull_fvgs  = df["fvg_bull"].sum()
    bear_fvgs  = df["fvg_bear"].sum()
    bull_obs   = df["ob_bull"].sum()
    bear_obs   = df["ob_bear"].sum()
    bull_sweeps= df["bull_liq_sweep"].sum()
    bear_sweeps= df["bear_liq_sweep"].sum()
    bos_count  = (df["bos"] != 0).sum()
    choch_count= (df["choch"] != 0).sum()

    print(f"\n  Summary for {interval}:")
    print(f"    FVGs:           {bull_fvgs} bullish / {bear_fvgs} bearish")
    print(f"    Order Blocks:   {bull_obs} bullish / {bear_obs} bearish")
    print(f"    Liq Sweeps:     {bull_sweeps} bullish / {bear_sweeps} bearish")
    print(f"    BOS events:     {bos_count}")
    print(f"    CHoCH events:   {choch_count}")
    print(f"    Total columns:  {len(df.columns)}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"ICT Enrichment — {SYMBOL}")
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")

    for interval in INTERVALS:
        try:
            enrich(interval)
        except Exception as e:
            import traceback
            print(f"\n  ERROR on {interval}: {e}")
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"  All done. Enriched files saved to: {OUTPUT_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
