"""
enrich_ict_v3.py
ICT Feature Enrichment - Version 3

Changes from v2:
  FIX 1: DST-aware session labels (uses zoneinfo, correct local time)
  FIX 2: Tightened OB definition (requires BOS, not just ATR move)
  FIX 3: Market structure uses BOTH wick and close versions
  FIX 4: Full multi-FVG tracking (nearest, oldest, above/below counts)
  FIX 5: CVD - session reset + 7d rolling + z-score
  FIX 6: Adaptive swing lookback based on ATR percentile
  FIX 7: Improved displacement flags (directional, gap-leaving, consecutive)
  NEW:   ny_open_830 fully correct on all timeframes (from v2)
  NEW:   ndog/nwog no-gap zeros (from v2)
  NEW:   All v2 extended features retained

Requires: pip install pandas numpy
Python 3.9+ required for zoneinfo (built-in). Earlier versions: pip install backports.zoneinfo
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timezone

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

import warnings
warnings.filterwarnings("ignore", message="Converting to PeriodArray")

# ==============================================================================
# CONFIG
# ==============================================================================

INPUT_DIR  = r"C:\Users\tjall\Desktop\Trading\data"
OUTPUT_DIR = r"C:\Users\tjall\Desktop\Trading\data"
SYMBOL     = "BTCUSDT"
START_DATE = "2017-08-17"
END_DATE   = "2026-03-01"

SWING_LOOKBACK_BASE = {
    "5m": 10, "15m": 8, "30m": 6,
    "1h": 5,  "4h":  4, "1d":  3,
}
SWING_LOOKBACK_MIN = {"5m": 3,  "15m": 3, "30m": 3, "1h": 3, "4h": 3, "1d": 3}
SWING_LOOKBACK_MAX = {"5m": 20, "15m": 16,"30m": 12,"1h": 10,"4h": 8, "1d": 6}

OB_ATR_MULTIPLIER     = 1.5
OB_MAX_BOS_LOOKFORWARD = 20   # candles to look forward for BOS after OB candidate
FVG_MIN_ATR_FRACTION  = 0.1
OTE_LOW               = 0.62
OTE_HIGH              = 0.79
DEALING_RANGE_MIN_ATR = 1.0
VOLUME_PCT_WINDOW     = 100
DISPLACEMENT_ATR_MULT = 2.0

INTERVAL_MS = {
    "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000,
}
INTERVAL_MINS = {
    "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "4h": 240, "1d": 1440,
}

NY_TZ  = ZoneInfo("America/New_York")
LON_TZ = ZoneInfo("Europe/London")

# ==============================================================================
# LOAD / SAVE
# ==============================================================================

def load_csv(interval):
    path = os.path.join(INPUT_DIR,
        f"{SYMBOL}_BINANCE_{interval}_{START_DATE}_to_{END_DATE}.csv")
    if not os.path.exists(path):
        print(f"  File not found: {path}")
        return None
    df = pd.read_csv(path, low_memory=False)
    df["open_time_utc"] = pd.to_datetime(df["open_time_utc"], utc=True)
    df["open_time"]     = df["open_time"].astype(np.int64)
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)
    df = df.sort_values("open_time").reset_index(drop=True)
    print(f"  Loaded {len(df):,} rows for {interval}")
    return df

def save_csv(df, interval):
    filename = f"{SYMBOL}_BINANCE_{interval}_{START_DATE}_to_{END_DATE}_enriched_v3.csv"
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, index=False)
    print(f"  Saved {len(df):,} rows -> {filename}")

# ==============================================================================
# FIX 1: DST-AWARE SESSION LABELS
# ==============================================================================

def add_session_labels_dst(df):
    """
    Killzone hours in LOCAL time (DST-aware):
      Asia:   20:00-00:00 NY time (previous evening)
      London: 02:00-05:00 NY time
      NY AM:  07:00-10:00 NY time (main killzone)

    Convert UTC -> NY local time for each row, then apply rules.
    This correctly handles EST (UTC-5) vs EDT (UTC-4) transitions.
    """
    # Convert to NY local time
    ny_times   = df["open_time_utc"].dt.tz_convert(NY_TZ)
    ny_hour    = ny_times.dt.hour
    ny_minute  = ny_times.dt.minute
    ny_mins    = ny_hour * 60 + ny_minute

    # Killzone windows in NY local minutes
    # Asia:   20:00-00:00 = 1200-1440 mins
    # London: 02:00-05:00 = 120-300 mins
    # NY AM:  07:00-10:00 = 420-600 mins
    conditions = [
        (ny_mins >= 1200) | (ny_mins < 60),   # 20:00-01:00 NY = Asia session
        (ny_mins >= 120)  & (ny_mins < 300),   # 02:00-05:00 NY = London killzone
        (ny_mins >= 420)  & (ny_mins < 600),   # 07:00-10:00 NY = NY AM killzone
    ]
    df["session"] = np.select(conditions, ["Asia","London","NewYork"], default="Off")

    # Also store the actual NY hour for reference
    df["ny_hour"] = ny_hour.values.astype(np.int8)

    return df

# ==============================================================================
# FIX 6: ADAPTIVE SWING LOOKBACK
# ==============================================================================

def compute_adaptive_lookback(atr_values, interval,
                               window=100,
                               base=None, lo=None, hi=None):
    """
    Lookback scales inversely with ATR percentile:
    - High volatility (ATR in 80th pct) -> lower lookback (swings form fast)
    - Low volatility  (ATR in 20th pct) -> higher lookback (need more candles)
    """
    base = base or SWING_LOOKBACK_BASE[interval]
    lo   = lo   or SWING_LOOKBACK_MIN[interval]
    hi   = hi   or SWING_LOOKBACK_MAX[interval]

    atr_s   = pd.Series(atr_values)
    atr_pct = atr_s.rolling(window, min_periods=10).rank(pct=True).fillna(0.5)

    # Invert: high ATR pct -> low lookback
    lookback = (hi - (atr_pct * (hi - lo))).round().astype(int).clip(lo, hi)
    return lookback.values

# ==============================================================================
# SWINGS (adaptive lookback)
# ==============================================================================

def add_swings_adaptive(df, interval):
    """
    Swing detection with per-candle adaptive lookback.
    Uses wick highs/lows (correct ICT definition).
    """
    highs = df["high"].values
    lows  = df["low"].values
    atr   = df["atr_14"].values if "atr_14" in df.columns else np.ones(len(df))
    size  = len(df)

    lookbacks = compute_adaptive_lookback(atr, interval)

    swing_high_flag  = np.zeros(size, dtype=np.int8)
    swing_low_flag   = np.zeros(size, dtype=np.int8)
    swing_high_price = np.full(size, np.nan)
    swing_low_price  = np.full(size, np.nan)

    for i in range(size):
        n = int(lookbacks[i])
        if i < n or i >= size - n:
            continue
        wh = highs[i-n : i+n+1]
        wl = lows[i-n  : i+n+1]
        if highs[i] == np.max(wh):
            swing_high_flag[i]  = 1
            swing_high_price[i] = highs[i]
        if lows[i] == np.min(wl):
            swing_low_flag[i]  = 1
            swing_low_price[i] = lows[i]

    df["swing_high"]       = swing_high_flag
    df["swing_low"]        = swing_low_flag
    df["swing_high_price"] = swing_high_price
    df["swing_low_price"]  = swing_low_price
    df["swing_lookback"]   = lookbacks.astype(np.int8)
    return df

# ==============================================================================
# FIX 3: MARKET STRUCTURE - BOTH WICK AND CLOSE VERSIONS
# ==============================================================================

def add_market_structure_v3(df):
    """
    Two versions of BOS/CHoCH:
      _close: close must exceed swing level (conservative, fewer false breaks)
      _wick:  high/low wick must exceed swing level (sensitive, more signals)

    market_trend uses the close version as the primary trend column.
    """
    size   = len(df)
    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values
    sh_p   = df["swing_high_price"].values
    sl_p   = df["swing_low_price"].values
    sh     = df["swing_high"].values
    sl     = df["swing_low"].values

    for version, use_wick in [("close", False), ("wick", True)]:
        bos   = np.zeros(size, dtype=np.int8)
        choch = np.zeros(size, dtype=np.int8)
        trend = np.zeros(size, dtype=np.int8)

        last_sh = last_sl = np.nan
        current_trend = 0

        for i in range(size):
            if sh[i] == 1 and not np.isnan(sh_p[i]):
                last_sh = sh_p[i]
            if sl[i] == 1 and not np.isnan(sl_p[i]):
                last_sl = sl_p[i]

            # Break level: close vs wick
            break_up   = closes[i] > last_sh if not use_wick else highs[i] > last_sh
            break_down = closes[i] < last_sl if not use_wick else lows[i]  < last_sl

            if not np.isnan(last_sh) and break_up:
                if current_trend == 1:
                    bos[i] = 1
                elif current_trend == -1:
                    choch[i] = 1
                current_trend = 1
            elif not np.isnan(last_sl) and break_down:
                if current_trend == -1:
                    bos[i] = -1
                elif current_trend == 1:
                    choch[i] = -1
                current_trend = -1

            trend[i] = current_trend

        suffix = f"_{version}"
        df[f"bos{suffix}"]          = bos
        df[f"choch{suffix}"]        = choch
        df[f"market_trend{suffix}"] = trend

    # Primary columns (close version) kept with original names for compatibility
    df["bos"]          = df["bos_close"]
    df["choch"]        = df["choch_close"]
    df["market_trend"] = df["market_trend_close"]

    return df

# ==============================================================================
# FIX 4: FULL MULTI-FVG TRACKING
# ==============================================================================

def add_fvg_v3(df, min_atr_fraction=FVG_MIN_ATR_FRACTION):
    """
    Tracks ALL active unmitigated FVGs simultaneously.
    For each candle stores:
      - Most recent unmitigated FVG (what you'd trade into next)
      - Nearest FVG to current price (most likely tested next)
      - Oldest active unmitigated FVG (strong magnet levels)
      - Count of active FVGs above / below current price separately
      - Total active count (confluence)

    Mitigation: price touches 50% midpoint (ICT equilibrium).
    """
    size  = len(df)
    highs = df["high"].values
    lows  = df["low"].values
    closes= df["close"].values
    atr   = df["atr_14"].values if "atr_14" in df.columns else np.ones(size)

    # --- Detect formation candles (unchanged logic) ---
    bull_flag = np.zeros(size, dtype=np.int8)
    bear_flag = np.zeros(size, dtype=np.int8)
    bull_top_raw = np.full(size, np.nan)
    bull_bot_raw = np.full(size, np.nan)
    bear_top_raw = np.full(size, np.nan)
    bear_bot_raw = np.full(size, np.nan)

    for i in range(1, size - 1):
        gap_bull = lows[i+1]  - highs[i-1]
        gap_bear = lows[i-1]  - highs[i+1]
        min_size = atr[i] * min_atr_fraction if not np.isnan(atr[i]) else 0
        if gap_bull > min_size:
            bull_flag[i]     = 1
            bull_top_raw[i]  = lows[i+1]
            bull_bot_raw[i]  = highs[i-1]
        if gap_bear > min_size:
            bear_flag[i]     = 1
            bear_top_raw[i]  = lows[i-1]
            bear_bot_raw[i]  = highs[i+1]

    df["fvg_bull"] = bull_flag
    df["fvg_bear"] = bear_flag

    # --- Multi-FVG tracking ---
    # Recent = most recently formed unmitigated
    out = {
        "fvg_bull_recent_top": np.full(size, np.nan),
        "fvg_bull_recent_bot": np.full(size, np.nan),
        "fvg_bull_recent_mid": np.full(size, np.nan),
        "fvg_bull_recent_age": np.full(size, np.nan),
        "fvg_bull_recent_mit": np.zeros(size, dtype=np.int8),
        "fvg_bull_nearest_top": np.full(size, np.nan),
        "fvg_bull_nearest_bot": np.full(size, np.nan),
        "fvg_bull_oldest_top":  np.full(size, np.nan),
        "fvg_bull_oldest_bot":  np.full(size, np.nan),
        "fvg_bull_count_above": np.zeros(size, dtype=np.int16),
        "fvg_bull_count_below": np.zeros(size, dtype=np.int16),
        "fvg_bull_count_total": np.zeros(size, dtype=np.int16),

        "fvg_bear_recent_top": np.full(size, np.nan),
        "fvg_bear_recent_bot": np.full(size, np.nan),
        "fvg_bear_recent_mid": np.full(size, np.nan),
        "fvg_bear_recent_age": np.full(size, np.nan),
        "fvg_bear_recent_mit": np.zeros(size, dtype=np.int8),
        "fvg_bear_nearest_top": np.full(size, np.nan),
        "fvg_bear_nearest_bot": np.full(size, np.nan),
        "fvg_bear_oldest_top":  np.full(size, np.nan),
        "fvg_bear_oldest_bot":  np.full(size, np.nan),
        "fvg_bear_count_above": np.zeros(size, dtype=np.int16),
        "fvg_bear_count_below": np.zeros(size, dtype=np.int16),
        "fvg_bear_count_total": np.zeros(size, dtype=np.int16),
    }

    # active_fvg: list of dicts {top, bot, mid, age, formed_at}
    active_bull = []
    active_bear = []

    for i in range(size):
        # Register new FVGs
        if bull_flag[i] == 1:
            mid = (bull_top_raw[i] + bull_bot_raw[i]) / 2
            active_bull.append({"top": bull_top_raw[i], "bot": bull_bot_raw[i],
                                 "mid": mid, "age": 0, "formed": i})
        if bear_flag[i] == 1:
            mid = (bear_top_raw[i] + bear_bot_raw[i]) / 2
            active_bear.append({"top": bear_top_raw[i], "bot": bear_bot_raw[i],
                                 "mid": mid, "age": 0, "formed": i})

        # Check mitigation and age all active FVGs
        # Age first, then check mitigation (skip candle of formation)
        for f in active_bull: f["age"] += 1
        for f in active_bear: f["age"] += 1

        active_bull = [f for f in active_bull
                       if not (f["age"] > 1 and lows[i] <= f["mid"])]
        active_bear = [f for f in active_bear
                       if not (f["age"] > 1 and highs[i] >= f["mid"])]

        c = closes[i]

        # Bull FVG stats
        if active_bull:
            recent  = active_bull[-1]
            oldest  = active_bull[0]
            above   = [f for f in active_bull if f["bot"] > c]
            below   = [f for f in active_bull if f["top"] < c]
            nearest = min(active_bull,
                          key=lambda f: min(abs(f["top"] - c), abs(f["bot"] - c)))

            out["fvg_bull_recent_top"][i] = recent["top"]
            out["fvg_bull_recent_bot"][i] = recent["bot"]
            out["fvg_bull_recent_mid"][i] = recent["mid"]
            out["fvg_bull_recent_age"][i] = recent["age"]
            out["fvg_bull_nearest_top"][i] = nearest["top"]
            out["fvg_bull_nearest_bot"][i] = nearest["bot"]
            out["fvg_bull_oldest_top"][i]  = oldest["top"]
            out["fvg_bull_oldest_bot"][i]  = oldest["bot"]
            out["fvg_bull_count_above"][i] = len(above)
            out["fvg_bull_count_below"][i] = len(below)
            out["fvg_bull_count_total"][i] = len(active_bull)

        # Bear FVG stats
        if active_bear:
            recent  = active_bear[-1]
            oldest  = active_bear[0]
            above   = [f for f in active_bear if f["bot"] > c]
            below   = [f for f in active_bear if f["top"] < c]
            nearest = min(active_bear,
                          key=lambda f: min(abs(f["top"] - c), abs(f["bot"] - c)))

            out["fvg_bear_recent_top"][i] = recent["top"]
            out["fvg_bear_recent_bot"][i] = recent["bot"]
            out["fvg_bear_recent_mid"][i] = recent["mid"]
            out["fvg_bear_recent_age"][i] = recent["age"]
            out["fvg_bear_nearest_top"][i] = nearest["top"]
            out["fvg_bear_nearest_bot"][i] = nearest["bot"]
            out["fvg_bear_oldest_top"][i]  = oldest["top"]
            out["fvg_bear_oldest_bot"][i]  = oldest["bot"]
            out["fvg_bear_count_above"][i] = len(above)
            out["fvg_bear_count_below"][i] = len(below)
            out["fvg_bear_count_total"][i] = len(active_bear)

    for col, arr in out.items():
        df[col] = arr

    # Keep legacy column names for master merge compatibility
    df["fvg_bull_top"]       = df["fvg_bull_recent_top"]
    df["fvg_bull_bot"]       = df["fvg_bull_recent_bot"]
    df["fvg_bull_mid"]       = df["fvg_bull_recent_mid"]
    df["fvg_bull_age"]       = df["fvg_bull_recent_age"]
    df["fvg_bull_mitigated"] = df["fvg_bull_recent_mit"]
    df["fvg_bear_top"]       = df["fvg_bear_recent_top"]
    df["fvg_bear_bot"]       = df["fvg_bear_recent_bot"]
    df["fvg_bear_mid"]       = df["fvg_bear_recent_mid"]
    df["fvg_bear_age"]       = df["fvg_bear_recent_age"]
    df["fvg_bear_mitigated"] = df["fvg_bear_recent_mit"]

    return df

# ==============================================================================
# FIX 2: TIGHTENED ORDER BLOCKS (require BOS)
# ==============================================================================

def add_order_blocks_v3(df, atr_mult=OB_ATR_MULTIPLIER,
                         max_bos_lookforward=OB_MAX_BOS_LOOKFORWARD):
    """
    Strict ICT OB definition:
      Bullish OB = last bearish candle before a bullish BOS (close_version)
      Bearish OB = last bullish candle before a bearish BOS (close_version)

    The OB must be the LAST opposing candle immediately before the move
    that causes the BOS - not just any candle before a large move.

    Also stores the BOS candle index so you can see how far the
    displacement was from the OB.
    """
    size   = len(df)
    opens  = df["open"].values
    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values
    atr    = df["atr_14"].values if "atr_14" in df.columns else np.ones(size)
    bos    = df["bos_close"].values if "bos_close" in df.columns else df["bos"].values

    bull_ob_flag = np.zeros(size, dtype=np.int8)
    bear_ob_flag = np.zeros(size, dtype=np.int8)
    bull_ob_top  = np.full(size, np.nan)
    bull_ob_bot  = np.full(size, np.nan)
    bear_ob_top  = np.full(size, np.nan)
    bear_ob_bot  = np.full(size, np.nan)

    for i in range(size - max_bos_lookforward):
        # Look forward for a bullish BOS within window
        future_bos = bos[i+1 : i+max_bos_lookforward+1]
        bull_bos_positions = np.where(future_bos == 1)[0]
        bear_bos_positions = np.where(future_bos == -1)[0]

        # Bullish OB: find last bearish candle before first bullish BOS
        if len(bull_bos_positions) > 0:
            bos_pos = bull_bos_positions[0]  # first bull BOS after i
            # Search backwards from BOS for last bearish candle
            for j in range(i + bos_pos, i, -1):
                if j >= size:
                    break
                if closes[j] < opens[j]:  # bearish candle
                    if not np.isnan(atr[j]) and atr[j] > 0:
                        bull_ob_flag[j] = 1
                        bull_ob_top[j]  = opens[j]   # body top = open of red candle
                        bull_ob_bot[j]  = closes[j]  # body bot = close of red candle
                    break  # only mark the LAST bearish candle

        # Bearish OB: find last bullish candle before first bearish BOS
        if len(bear_bos_positions) > 0:
            bos_pos = bear_bos_positions[0]
            for j in range(i + bos_pos, i, -1):
                if j >= size:
                    break
                if closes[j] > opens[j]:  # bullish candle
                    if not np.isnan(atr[j]) and atr[j] > 0:
                        bear_ob_flag[j] = 1
                        bear_ob_top[j]  = closes[j]  # body top = close of green candle
                        bear_ob_bot[j]  = opens[j]   # body bot = open of green candle
                    break

    df["ob_bull"] = bull_ob_flag
    df["ob_bear"] = bear_ob_flag

    # Forward-fill most recent active OB + mitigation tracking
    curr_bull_top = curr_bull_bot = np.nan
    curr_bear_top = curr_bear_bot = np.nan
    bull_mit = bear_mit = 0
    bull_formed = bear_formed = False
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
            curr_bull_top = bull_ob_top[i]
            curr_bull_bot = bull_ob_bot[i]
            bull_mit = 0; bull_formed = True; bull_age = 0
        if bear_ob_flag[i] == 1:
            curr_bear_top = bear_ob_top[i]
            curr_bear_bot = bear_ob_bot[i]
            bear_mit = 0; bear_formed = True; bear_age = 0

        if bull_formed and not bull_mit and not np.isnan(curr_bull_top):
            if lows[i] <= curr_bull_top and highs[i] >= curr_bull_bot:
                bull_mit = 1
        if bear_formed and not bear_mit and not np.isnan(curr_bear_top):
            if highs[i] >= curr_bear_bot and lows[i] <= curr_bear_top:
                bear_mit = 1

        if bull_formed:
            out_bull_top[i] = curr_bull_top
            out_bull_bot[i] = curr_bull_bot
            out_bull_mit[i] = bull_mit
            out_bull_age[i] = bull_age
            bull_age += 1
        if bear_formed:
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

# ==============================================================================
# FIX 5: IMPROVED CVD
# ==============================================================================

def add_volume_features_v3(df):
    """Buy/sell split, delta, CVD with three variants:
      cvd_daily   - resets at midnight UTC (intraday context)
      cvd_session - resets at each session boundary (session pressure)
      cvd_7d      - rolling 7-day non-resetting (trend context)
      cvd_zscore  - cvd_daily normalized by 20-period rolling std
    """
    df["buy_vol"]  = df["taker_base"].astype(float)
    df["sell_vol"] = df["volume"] - df["buy_vol"]
    df["delta"]    = df["buy_vol"] - df["sell_vol"]

    dt = df["open_time_utc"]

    # Daily CVD
    df["_date"] = dt.dt.date
    df["cvd_daily"] = df.groupby("_date")["delta"].cumsum()
    df["cvd"] = df["cvd_daily"]   # keep legacy name

    # Session CVD
    df["_session_group"] = (df["session"] != df["session"].shift(1)).cumsum()
    df["cvd_session"] = df.groupby("_session_group")["delta"].cumsum()

    # Rolling 7-day CVD (non-resetting)
    # Approximate as rolling sum of delta over last 7*288 = 2016 candles on 5m
    # For other timeframes calculate proportionally
    df["cvd_7d"] = df["delta"].rolling(2016, min_periods=1).sum()

    # CVD z-score (daily CVD normalized)
    cvd_std = df["cvd_daily"].rolling(20, min_periods=5).std()
    df["cvd_zscore"] = np.where(
        cvd_std > 0,
        df["cvd_daily"] / cvd_std,
        0.0
    )

    df.drop(columns=["_date", "_session_group"], inplace=True)
    return df

# ==============================================================================
# FIX 7: IMPROVED DISPLACEMENT FLAGS
# ==============================================================================

def add_displacement_v3(df, atr_mult=DISPLACEMENT_ATR_MULT):
    """
    Directional displacement flags:
      displacement_bull  - large green candle, closes in upper 25% of range, > 2x ATR
      displacement_bear  - large red candle, closes in lower 25% of range, > 2x ATR
      displacement_gap   - displacement that also leaves an FVG
      displacement_consec- 2+ consecutive displacements same direction
      displacement_any   - either direction (legacy)
    """
    if "atr_14" not in df.columns:
        for col in ["displacement_bull","displacement_bear",
                    "displacement_gap","displacement_consec","displacement_any"]:
            df[col] = 0
        return df

    atr    = df["atr_14"].values
    opens  = df["open"].values
    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values
    fvg_b  = df["fvg_bull"].values if "fvg_bull" in df.columns else np.zeros(len(df))
    fvg_r  = df["fvg_bear"].values if "fvg_bear" in df.columns else np.zeros(len(df))
    size   = len(df)

    disp_bull   = np.zeros(size, dtype=np.int8)
    disp_bear   = np.zeros(size, dtype=np.int8)
    disp_gap    = np.zeros(size, dtype=np.int8)
    disp_consec = np.zeros(size, dtype=np.int8)

    for i in range(size):
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue
        rng       = highs[i] - lows[i]
        threshold = atr[i] * atr_mult
        if rng < threshold:
            continue

        is_bull   = closes[i] > opens[i]
        is_bear   = closes[i] < opens[i]
        close_pos = (closes[i] - lows[i]) / rng if rng > 0 else 0.5

        if is_bull and close_pos >= 0.75:  # closes in upper 25%
            disp_bull[i] = 1
        if is_bear and close_pos <= 0.25:  # closes in lower 25%
            disp_bear[i] = 1

    # Gap flag: displacement that also created an FVG
    for i in range(size):
        if (disp_bull[i] or disp_bear[i]) and (fvg_b[i] or fvg_r[i]):
            disp_gap[i] = 1

    # Consecutive: 2+ same-direction displacements in a row
    for i in range(1, size):
        if (disp_bull[i] and disp_bull[i-1]) or (disp_bear[i] and disp_bear[i-1]):
            disp_consec[i] = 1

    df["displacement_bull"]   = disp_bull
    df["displacement_bear"]   = disp_bear
    df["displacement_gap"]    = disp_gap
    df["displacement_consec"] = disp_consec
    df["displacement_any"]    = ((disp_bull + disp_bear) > 0).astype(np.int8)
    return df

# ==============================================================================
# RETAINED FROM V2 (unchanged)
# ==============================================================================

def add_atr(df, period=14):
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h-l),(h-pc).abs(),(l-pc).abs()], axis=1).max(axis=1)
    df[f"atr_{period}"] = tr.ewm(alpha=1/period, min_periods=period).mean()
    return df

def add_realized_vol(df, period=20):
    df[f"realized_vol_{period}"] = np.log(df["close"]/df["close"].shift(1)).rolling(period).std()
    return df

def add_reference_prices(df, interval):
    df = df.copy()
    dt = df["open_time_utc"]
    df["_date"]     = dt.dt.date
    df["_utc_mins"] = dt.dt.hour * 60 + dt.dt.minute
    itvl = INTERVAL_MINS.get(interval, 5)

    df["midnight_open"] = np.where(df["_utc_mins"] == 0, df["open"], np.nan)
    df["midnight_open"] = df.groupby("_date")["midnight_open"].transform("ffill")

    df["_candle_end"] = df["_utc_mins"] + itvl
    contains_830 = (df["_utc_mins"] <= 810) & (df["_candle_end"] >= 810)
    df["ny_open_830"] = np.where(contains_830, df["open"], np.nan)
    df["ny_open_830"] = df.groupby("_date")["ny_open_830"].transform("ffill")

    df.drop(columns=["_date","_utc_mins","_candle_end"], inplace=True)
    return df

def add_previous_levels(df):
    df = df.copy()
    dt = df["open_time_utc"]
    df["_date"]  = dt.dt.date
    df["_week"]  = dt.dt.to_period("W")
    df["_month"] = dt.dt.to_period("M")

    daily = df.groupby("_date").agg(dh=("high","max"),dl=("low","min"),dc=("close","last")).reset_index()
    daily["pdh"]=daily["dh"].shift(1); daily["pdl"]=daily["dl"].shift(1); daily["pdc"]=daily["dc"].shift(1)
    df = df.merge(daily[["_date","pdh","pdl","pdc"]], on="_date", how="left")

    weekly = df.groupby("_week").agg(wh=("high","max"),wl=("low","min"),wc=("close","last")).reset_index()
    weekly["pwh"]=weekly["wh"].shift(1); weekly["pwl"]=weekly["wl"].shift(1); weekly["pwc"]=weekly["wc"].shift(1)
    df = df.merge(weekly[["_week","pwh","pwl","pwc"]], on="_week", how="left")

    monthly = df.groupby("_month").agg(mh=("high","max"),ml=("low","min"),mc=("close","last")).reset_index()
    monthly["pmh"]=monthly["mh"].shift(1); monthly["pml"]=monthly["ml"].shift(1); monthly["pmc"]=monthly["mc"].shift(1)
    df = df.merge(monthly[["_month","pmh","pml","pmc"]], on="_month", how="left")

    df.drop(columns=["_date","_week","_month"], inplace=True)
    return df

def add_liquidity_sweeps(df):
    size   = len(df)
    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values
    sh     = df["swing_high"].values
    sl     = df["swing_low"].values
    sh_p   = df["swing_high_price"].values
    sl_p   = df["swing_low_price"].values

    bull_sweep  = np.zeros(size, dtype=np.int8)
    bear_sweep  = np.zeros(size, dtype=np.int8)
    sweep_level = np.full(size, np.nan)
    intact_highs = []
    intact_lows  = []

    for i in range(size):
        if sh[i]==1 and not np.isnan(sh_p[i]): intact_highs.append((i, sh_p[i]))
        if sl[i]==1 and not np.isnan(sl_p[i]): intact_lows.append((i, sl_p[i]))

        swept_h = [(idx,p) for idx,p in intact_highs if highs[i]>p and closes[i]<p]
        for idx,p in swept_h: bear_sweep[i]=1; sweep_level[i]=p
        intact_highs = [(idx,p) for idx,p in intact_highs if (idx,p) not in swept_h]

        swept_l = [(idx,p) for idx,p in intact_lows if lows[i]<p and closes[i]>p]
        for idx,p in swept_l: bull_sweep[i]=1; sweep_level[i]=p
        intact_lows = [(idx,p) for idx,p in intact_lows if (idx,p) not in swept_l]

    df["bull_liq_sweep"] = bull_sweep
    df["bear_liq_sweep"] = bear_sweep
    df["sweep_level"]    = sweep_level
    return df

def add_dealing_range(df):
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

    dr_high=np.full(size,np.nan); dr_low=np.full(size,np.nan); dr_eq=np.full(size,np.nan)
    premium=np.zeros(size,dtype=np.int8); discount=np.zeros(size,dtype=np.int8)
    ote_zone=np.zeros(size,dtype=np.int8)
    last_sh=last_sl=np.nan

    for i in range(size):
        if sh[i]==1 and not np.isnan(sh_p[i]): last_sh=sh_p[i]
        if sl[i]==1 and not np.isnan(sl_p[i]): last_sl=sl_p[i]
        if np.isnan(last_sh) or np.isnan(last_sl): continue
        rng = last_sh - last_sl
        if np.isnan(atr[i]) or rng < atr[i] * DEALING_RANGE_MIN_ATR: continue
        eq = last_sl + rng*0.5
        dr_high[i]=last_sh; dr_low[i]=last_sl; dr_eq[i]=eq
        premium[i]=1 if closes[i]>eq else 0
        discount[i]=1 if closes[i]<=eq else 0
        if trend[i]==1:
            if (last_sh-rng*OTE_HIGH)<=closes[i]<=(last_sh-rng*OTE_LOW): ote_zone[i]=1
        elif trend[i]==-1:
            if (last_sl+rng*OTE_LOW)<=closes[i]<=(last_sl+rng*OTE_HIGH): ote_zone[i]=1

    df["dr_high"]=dr_high; df["dr_low"]=dr_low; df["dr_eq"]=dr_eq
    df["premium"]=premium; df["discount"]=discount; df["ote_zone"]=ote_zone
    return df

def add_opening_gaps(df):
    df = df.copy()
    dt = df["open_time_utc"]
    df["_date"] = dt.dt.date
    df["_week"] = dt.dt.to_period("W")

    daily_close = df.groupby("_date")["close"].last().reset_index()
    daily_close.columns = ["_date","_pdc"]
    daily_close["_pdc"] = daily_close["_pdc"].shift(1)
    df = df.merge(daily_close, on="_date", how="left")

    daily_open = df.groupby("_date")["open"].first().reset_index()
    daily_open.columns = ["_date","_dopen"]
    df = df.merge(daily_open, on="_date", how="left")

    gap_exists = df["_dopen"] != df["_pdc"]
    df["ndog_high"]   = np.where(gap_exists, np.maximum(df["_dopen"],df["_pdc"]), df["_dopen"])
    df["ndog_low"]    = np.where(gap_exists, np.minimum(df["_dopen"],df["_pdc"]), df["_dopen"])
    df["ndog_exists"] = gap_exists.astype(np.int8)

    weekly_close = df.groupby("_week")["close"].last().reset_index()
    weekly_close.columns = ["_week","_pwc"]
    weekly_close["_pwc"] = weekly_close["_pwc"].shift(1)
    df = df.merge(weekly_close, on="_week", how="left")

    weekly_open = df.groupby("_week")["open"].first().reset_index()
    weekly_open.columns = ["_week","_wopen"]
    df = df.merge(weekly_open, on="_week", how="left")

    wgap = df["_wopen"] != df["_pwc"]
    df["nwog_high"]   = np.where(wgap, np.maximum(df["_wopen"],df["_pwc"]), df["_wopen"])
    df["nwog_low"]    = np.where(wgap, np.minimum(df["_wopen"],df["_pwc"]), df["_wopen"])
    df["nwog_exists"] = wgap.astype(np.int8)

    for col in ["ndog_high","ndog_low","ndog_exists"]:
        df[col] = df.groupby("_date")[col].transform("ffill")
    for col in ["nwog_high","nwog_low","nwog_exists"]:
        df[col] = df.groupby("_week")[col].transform("ffill")

    df.drop(columns=["_date","_week","_pdc","_dopen","_pwc","_wopen"], inplace=True)
    return df

def add_distance_to_levels(df):
    close = df["close"]
    for col,label in [("pdh","pdh"),("pdl","pdl"),("pwh","pwh"),("pwl","pwl")]:
        if col in df.columns:
            diff = close - df[col]
            df[f"dist_{label}_dollars"] = diff
            df[f"dist_{label}_pct"]     = diff / df[col] * 100
    return df

def add_liquidity_levels(df):
    size   = len(df)
    closes = df["close"].values
    sh     = df["swing_high"].values
    sl     = df["swing_low"].values
    sh_p   = df["swing_high_price"].values
    sl_p   = df["swing_low_price"].values

    liq_above = np.full(size, np.nan)
    liq_below = np.full(size, np.nan)
    intact_highs = []
    intact_lows  = []

    for i in range(size):
        if sh[i]==1 and not np.isnan(sh_p[i]): intact_highs.append(sh_p[i])
        if sl[i]==1 and not np.isnan(sl_p[i]): intact_lows.append(sl_p[i])

        above = [p for p in intact_highs if p > closes[i]]
        below = [p for p in intact_lows  if p < closes[i]]
        if above: liq_above[i] = min(above)
        if below: liq_below[i] = max(below)

        intact_highs = [p for p in intact_highs if p > closes[i]]
        intact_lows  = [p for p in intact_lows  if p < closes[i]]

    df["liquidity_above"]        = liq_above
    df["liquidity_below"]        = liq_below
    df["dist_liq_above_dollars"] = liq_above - closes
    df["dist_liq_below_dollars"] = closes - liq_below
    df["dist_liq_above_pct"]     = (liq_above - closes) / closes * 100
    df["dist_liq_below_pct"]     = (closes - liq_below) / closes * 100
    return df

def add_volume_percentile(df, window=VOLUME_PCT_WINDOW):
    df["volume_percentile"] = df["volume"].rolling(window).rank(pct=True) * 100
    return df

def add_candles_since_choch(df):
    choch  = df["choch_close"].values if "choch_close" in df.columns else df["choch"].values
    size   = len(df)
    result = np.full(size, np.nan)
    last   = -1
    for i in range(size):
        if choch[i] != 0: last = i
        if last >= 0: result[i] = i - last
    df["candles_since_choch"] = result
    return df

def add_time_features(df):
    dt = df["open_time_utc"]
    df["hour_of_day"]  = dt.dt.hour.astype(np.int8)
    df["time_of_week"] = dt.dt.dayofweek.astype(np.int8)
    df["day_of_month"] = dt.dt.day.astype(np.int8)
    df["hour_sin"] = np.sin(2*np.pi*df["hour_of_day"]/24)
    df["hour_cos"] = np.cos(2*np.pi*df["hour_of_day"]/24)
    df["dow_sin"]  = np.sin(2*np.pi*df["time_of_week"]/7)
    df["dow_cos"]  = np.cos(2*np.pi*df["time_of_week"]/7)
    return df

def add_session_vwap(df):
    df = df.copy()
    df["_sg"] = (df["session"] != df["session"].shift(1)).cumsum()
    tp = (df["high"] + df["low"] + df["close"]) / 3
    df["_tpv"] = tp * df["volume"]
    df["session_vwap"] = (df.groupby("_sg")["_tpv"].cumsum() /
                          df.groupby("_sg")["volume"].cumsum())
    df["price_vs_vwap"] = np.sign(df["close"] - df["session_vwap"]).fillna(0).astype(np.int8)
    df.drop(columns=["_sg","_tpv"], inplace=True)
    return df

def add_funding_rate_features(df):
    if "funding_rate" not in df.columns or df["funding_rate"].isnull().all():
        df["funding_rate_24h_cum"] = np.nan
        df["funding_rate_zscore"]  = np.nan
        return df
    fr = df["funding_rate"]
    df["funding_rate_24h_cum"] = fr.rolling(3, min_periods=1).sum()
    fr_std = fr.rolling(30, min_periods=5).std()
    fr_mean= fr.rolling(30, min_periods=5).mean()
    df["funding_rate_zscore"] = np.where(
        fr_std > 0, (fr - fr_mean) / fr_std, 0.0
    )
    return df

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

INTERVALS = ["5m","15m","30m","1h","4h","1d"]

def enrich(interval):
    print(f"\n{'='*60}")
    print(f"  {SYMBOL} | {interval} - v3")
    print(f"{'='*60}")

    df = load_csv(interval)
    if df is None:
        return

    total = 21
    def step(n, name, fn):
        print(f"  [{n:02d}/{total}] {name}...")
        return fn()

    df = step(1,  "ATR",                  lambda: add_atr(df))
    df = step(2,  "Realized vol",         lambda: add_realized_vol(df))
    df = step(3,  "Session labels DST",   lambda: add_session_labels_dst(df))
    df = step(4,  "Volume + CVD v3",      lambda: add_volume_features_v3(df))
    df = step(5,  "Reference prices",     lambda: add_reference_prices(df, interval))
    df = step(6,  "Previous levels",      lambda: add_previous_levels(df))
    df = step(7,  "Swings (adaptive)",    lambda: add_swings_adaptive(df, interval))
    df = step(8,  "Liquidity sweeps",     lambda: add_liquidity_sweeps(df))
    df = step(9,  "Market structure v3",  lambda: add_market_structure_v3(df))
    df = step(10, "FVG v3 (multi-track)", lambda: add_fvg_v3(df))
    df = step(11, "Order Blocks v3",      lambda: add_order_blocks_v3(df))
    df = step(12, "Dealing range",        lambda: add_dealing_range(df))
    df = step(13, "Opening gaps",         lambda: add_opening_gaps(df))
    df = step(14, "Distance to levels",   lambda: add_distance_to_levels(df))
    df = step(15, "Liquidity levels",     lambda: add_liquidity_levels(df))
    df = step(16, "Volume percentile",    lambda: add_volume_percentile(df))
    df = step(17, "Displacement v3",      lambda: add_displacement_v3(df))
    df = step(18, "Candles since CHoCH",  lambda: add_candles_since_choch(df))
    df = step(19, "Time features",        lambda: add_time_features(df))
    df = step(20, "Session VWAP",         lambda: add_session_vwap(df))
    df = step(21, "Funding rate",         lambda: add_funding_rate_features(df))

    print(f"  Saving...")
    save_csv(df, interval)

    # Summary
    ny_null_pct = df["ny_open_830"].isnull().mean() * 100
    ob_bull     = df["ob_bull"].sum()
    ob_bear     = df["ob_bear"].sum()
    fvg_total   = df["fvg_bull_count_total"].mean()
    disp        = df["displacement_any"].sum()
    sess_counts = df["session"].value_counts().to_dict()

    print(f"\n  Summary for {interval}:")
    print(f"    Total columns:      {len(df.columns)}")
    print(f"    ny_open_830 nulls:  {ny_null_pct:.1f}% (should be ~0% on all TFs)")
    print(f"    Order Blocks:       {ob_bull} bull / {ob_bear} bear (v3 strict)")
    print(f"    Avg active FVGs:    {fvg_total:.2f}")
    print(f"    Displacement candles: {disp} ({disp/len(df)*100:.2f}%)")
    print(f"    Session distribution: {sess_counts}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"ICT Enrichment v3 -- {SYMBOL}")
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"DST-aware sessions: YES")
    print(f"Strict OB (requires BOS): YES")
    print(f"Multi-FVG tracking: YES")
    print(f"Both wick+close BOS: YES")

    for interval in INTERVALS:
        try:
            enrich(interval)
        except Exception as e:
            import traceback
            print(f"\n  ERROR on {interval}: {e}")
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"  Done. Files saved with suffix _enriched_v3.csv")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
