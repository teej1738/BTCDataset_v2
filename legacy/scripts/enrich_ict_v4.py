"""
enrich_ict_v4.py
ICT Feature Enrichment -- Version 4

Ported from enrich_ict_v3.py with v2 dataset conventions:
  - Reads parquet from data/resampled/, writes to data/enriched/
  - New column names: ict_ / sess_ / cvd_ / liq_ prefixes
  - Correct volume column names: volume_base, taker_buy_base
  - DST-aware Silver Bullet: three separate flags (not one combined)
  - Perp-specific columns (mark/index/basis/funding) handled gracefully
  - meta_is_complete carried through unchanged

Changes from v3:
  NEW: Silver Bullet flags split into sess_sb_london / sess_sb_ny_am / sess_sb_ny_pm
  NEW: Perp-aware CVD (taker_buy_base instead of taker_base)
  NEW: Silver Bullet session windows added to session label step
  NEW: All column names use v2 prefix convention (ict_ / sess_ / liq_ / cvd_)
  NEW: Input/output are parquet, not CSV
  NEW: Instrument-aware (spot vs perp)
  RETAINED: All v3 logic exactly as-is (swings, FVG, OB, structure, displacement)
  RETAINED: Adaptive swing lookback
  RETAINED: Multi-FVG tracking (nearest/oldest/recent/count)
  RETAINED: Strict OB (requires BOS)
  RETAINED: CVD daily/session/7d/zscore reset variants
  RETAINED: DST-aware session labels via zoneinfo

Requires: pip install pandas numpy pyarrow
Python 3.9+ required for zoneinfo (built-in).
Earlier versions: pip install backports.zoneinfo

Design decisions: see BTCDataset_v2/STRATEGY_LOG.md
"""

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timezone

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ==============================================================================
# CONFIG
# ==============================================================================

BASE_DIR    = r"C:\Users\tjall\Desktop\Trading\BTCDataset_v2"
INPUT_DIR   = os.path.join(BASE_DIR, "data", "resampled")
OUTPUT_DIR  = os.path.join(BASE_DIR, "data", "enriched")
SYMBOL      = "BTCUSDT"

# Instruments to enrich (spot uses spot resampled files, perp uses perp)
INSTRUMENTS = ["perp", "spot"]
INTERVALS   = ["5m", "15m", "30m", "1h", "4h", "1d"]

# Swing lookback parameters (adaptive, based on ATR percentile)
SWING_LOOKBACK_BASE = {"5m": 10, "15m": 8, "30m": 6, "1h": 5, "4h": 4, "1d": 3}
SWING_LOOKBACK_MIN  = {"5m": 3,  "15m": 3, "30m": 3, "1h": 3, "4h": 3, "1d": 3}
SWING_LOOKBACK_MAX  = {"5m": 20, "15m": 16,"30m": 12,"1h": 10,"4h": 8, "1d": 6}

OB_ATR_MULTIPLIER      = 1.5
OB_MAX_BOS_LOOKFORWARD = 20    # candles to look forward for BOS
FVG_MIN_ATR_FRACTION   = 0.1
# Lookback cap: FVGs older than this many bars are dropped regardless of mitigation.
# Set to ~1 trading day per timeframe to keep only contextually relevant FVGs.
FVG_LOOKBACK_BARS = {
    "5m":  288,   # 1 day
    "15m":  96,   # 1 day
    "30m":  48,   # 1 day
    "1h":   48,   # 2 days
    "4h":   30,   # 5 days
    "1d":   20,   # 4 weeks
}
FVG_LOOKBACK_DEFAULT = 96  # fallback if interval not in dict
OTE_LOW                = 0.62
OTE_HIGH               = 0.79
DEALING_RANGE_MIN_ATR  = 1.0
VOLUME_PCT_WINDOW      = 100
DISPLACEMENT_ATR_MULT  = 2.0

INTERVAL_MINS = {"5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}

NY_TZ  = ZoneInfo("America/New_York")
LON_TZ = ZoneInfo("Europe/London")

# ==============================================================================
# LOAD / SAVE
# ==============================================================================

def load_parquet(instrument, interval):
    fname = f"{SYMBOL}_{instrument}_{interval}.parquet"
    path  = os.path.join(INPUT_DIR, fname)
    if not os.path.exists(path):
        log.warning(f"  File not found: {path}")
        return None
    df = pd.read_parquet(path)
    df = df.sort_values("bar_start_ts_utc").reset_index(drop=True)
    log.info(f"  Loaded {len(df):,} rows  |  {len(df.columns)} cols  |  {interval}")
    return df

def save_parquet(df, instrument, interval):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fname = f"{SYMBOL}_{instrument}_{interval}_enriched.parquet"
    path  = os.path.join(OUTPUT_DIR, fname)
    df.to_parquet(path, index=False, engine="pyarrow")
    size_mb = os.path.getsize(path) / 1024**2
    log.info(f"  Saved: {fname}  ({size_mb:.1f} MB)  |  {len(df.columns)} cols")

# ==============================================================================
# STEP 1: ATR
# ==============================================================================

def add_atr(df, period=14):
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    df[f"ict_atr_{period}"] = tr.ewm(alpha=1 / period, min_periods=period).mean()
    return df

# ==============================================================================
# STEP 2: REALIZED VOLATILITY
# ==============================================================================

def add_realized_vol(df, period=20):
    df["ict_realized_vol_20"] = (
        np.log(df["close"] / df["close"].shift(1)).rolling(period).std()
    )
    return df

# ==============================================================================
# STEP 3: DST-AWARE SESSION LABELS + SILVER BULLET FLAGS
# ==============================================================================

def add_session_labels(df):
    """
    Killzone hours in NY LOCAL time (DST-aware via tz_convert).
    Uses bar_start_ts_utc which is datetime64[ns, UTC].

    Session windows (NY local):
      Asia:    20:00-01:00  (overnight carry from prior day)
      London:  02:00-05:00  (London open killzone)
      NY:      07:00-10:00  (NY AM killzone)

    Silver Bullet windows (NY local) -- three SEPARATE flags (D03 in STRATEGY_LOG):
      sess_sb_london:  03:00-04:00  (London Silver Bullet)
      sess_sb_ny_am:   10:00-11:00  (NY AM Silver Bullet)
      sess_sb_ny_pm:   14:00-15:00  (NY PM Silver Bullet)

    These are different setups. Do NOT combine them.
    """
    ny_times  = df["bar_start_ts_utc"].dt.tz_convert(NY_TZ)
    ny_hour   = ny_times.dt.hour
    ny_minute = ny_times.dt.minute
    ny_mins   = ny_hour * 60 + ny_minute

    # Session labels
    conditions = [
        (ny_mins >= 1200) | (ny_mins < 60),    # 20:00-01:00 NY = Asia
        (ny_mins >= 120)  & (ny_mins < 300),    # 02:00-05:00 NY = London
        (ny_mins >= 420)  & (ny_mins < 600),    # 07:00-10:00 NY = NY AM
    ]
    df["sess_label"]  = np.select(conditions, ["Asia", "London", "NewYork"], default="Off")
    df["sess_ny_hour"] = ny_hour.values.astype(np.int8)

    # Silver Bullet windows (D03: three separate flags, never combined)
    df["sess_sb_london"] = ((ny_mins >= 180) & (ny_mins < 240)).astype(np.int8)  # 03:00-04:00
    df["sess_sb_ny_am"]  = ((ny_mins >= 600) & (ny_mins < 660)).astype(np.int8)  # 10:00-11:00
    df["sess_sb_ny_pm"]  = ((ny_mins >= 840) & (ny_mins < 900)).astype(np.int8)  # 14:00-15:00

    return df

# ==============================================================================
# STEP 4: CVD -- RESET VARIANTS (never all-time cumsum)
# ==============================================================================

def add_cvd(df):
    """
    Real CVD using taker_buy_base (aggressor buys from Binance archive).
    Four reset variants per D07 in STRATEGY_LOG. Never all-time cumsum.

    Spot files have taker_buy_base.
    Perp files also have taker_buy_base.
    If column missing, skip gracefully (shouldn't happen with v2 data).
    """
    if "taker_buy_base" not in df.columns:
        log.warning("  taker_buy_base column missing, skipping CVD")
        for col in ["cvd_delta", "cvd_daily", "cvd_session", "cvd_7d", "cvd_zscore"]:
            df[col] = np.nan
        return df

    # Raw per-bar delta
    df["cvd_delta"] = 2.0 * df["taker_buy_base"] - df["volume_base"]

    # Daily CVD: reset at 00:00 UTC each day
    df["_date"] = df["bar_start_ts_utc"].dt.date
    df["cvd_daily"] = df.groupby("_date")["cvd_delta"].cumsum()

    # Session CVD: reset at each session boundary
    df["_session_group"] = (df["sess_label"] != df["sess_label"].shift(1)).cumsum()
    df["cvd_session"] = df.groupby("_session_group")["cvd_delta"].cumsum()

    # Rolling 7-day CVD: window in candles varies by interval
    # 5m: 7*288=2016, 15m: 7*96=672, 30m: 7*48=336, 1h: 7*24=168, 4h: 7*6=42, 1d: 7
    interval_mins = df.attrs.get("interval_mins", 5)
    candles_per_7d = max(1, int(7 * 24 * 60 / interval_mins))
    df["cvd_7d"] = df["cvd_delta"].rolling(candles_per_7d, min_periods=1).sum()

    # CVD z-score: daily CVD normalized by 20-period rolling std
    cvd_std = df["cvd_daily"].rolling(20, min_periods=5).std()
    df["cvd_zscore"] = np.where(cvd_std > 0, df["cvd_daily"] / cvd_std, 0.0)

    df.drop(columns=["_date", "_session_group"], inplace=True)
    return df

# ==============================================================================
# STEP 5: REFERENCE PRICES (midnight open, NY 8:30 open)
# ==============================================================================

def add_reference_prices(df, interval):
    dt   = df["bar_start_ts_utc"]
    itvl = INTERVAL_MINS.get(interval, 5)

    df["_date"]     = dt.dt.date
    df["_utc_mins"] = dt.dt.hour * 60 + dt.dt.minute

    # Midnight UTC open (00:00 bar open)
    df["ict_midnight_open"] = np.where(df["_utc_mins"] == 0, df["open"], np.nan)
    df["ict_midnight_open"] = df.groupby("_date")["ict_midnight_open"].transform("ffill")

    # NY 8:30 open (13:30 UTC in EDT, 14:30 UTC in EST -- use the candle that contains 13:30)
    # Detect via NY local time to handle DST correctly
    ny_times  = dt.dt.tz_convert(NY_TZ)
    ny_mins   = ny_times.dt.hour * 60 + ny_times.dt.minute
    ny_end    = ny_mins + itvl
    contains_830 = (ny_mins <= 510) & (ny_end > 510)  # 510 = 8:30 * 60 mins
    df["ict_ny_open_830"] = np.where(contains_830, df["open"], np.nan)
    df["ict_ny_open_830"] = df.groupby("_date")["ict_ny_open_830"].transform("ffill")

    df.drop(columns=["_date", "_utc_mins"], inplace=True)
    return df

# ==============================================================================
# STEP 6: PREVIOUS SESSION LEVELS (PDH, PDL, PWH, PWL, PMH, PML)
# ==============================================================================

def add_previous_levels(df):
    dt = df["bar_start_ts_utc"]
    df["_date"]  = dt.dt.date
    df["_week"]  = dt.dt.to_period("W")
    df["_month"] = dt.dt.to_period("M")

    daily = df.groupby("_date").agg(
        dh=("high", "max"), dl=("low", "min"), dc=("close", "last")
    ).reset_index()
    daily["liq_pdh"] = daily["dh"].shift(1)
    daily["liq_pdl"] = daily["dl"].shift(1)
    daily["liq_pdc"] = daily["dc"].shift(1)
    df = df.merge(daily[["_date", "liq_pdh", "liq_pdl", "liq_pdc"]], on="_date", how="left")

    weekly = df.groupby("_week").agg(
        wh=("high", "max"), wl=("low", "min"), wc=("close", "last")
    ).reset_index()
    weekly["liq_pwh"] = weekly["wh"].shift(1)
    weekly["liq_pwl"] = weekly["wl"].shift(1)
    weekly["liq_pwc"] = weekly["wc"].shift(1)
    df = df.merge(weekly[["_week", "liq_pwh", "liq_pwl", "liq_pwc"]], on="_week", how="left")

    monthly = df.groupby("_month").agg(
        mh=("high", "max"), ml=("low", "min"), mc=("close", "last")
    ).reset_index()
    monthly["liq_pmh"] = monthly["mh"].shift(1)
    monthly["liq_pml"] = monthly["ml"].shift(1)
    monthly["liq_pmc"] = monthly["mc"].shift(1)
    df = df.merge(monthly[["_month", "liq_pmh", "liq_pml", "liq_pmc"]], on="_month", how="left")

    df.drop(columns=["_date", "_week", "_month"], inplace=True)
    return df

# ==============================================================================
# STEP 7: ADAPTIVE SWING DETECTION
# ==============================================================================

def compute_adaptive_lookback(atr_values, interval, window=100):
    lo   = SWING_LOOKBACK_MIN[interval]
    hi   = SWING_LOOKBACK_MAX[interval]
    atr_s   = pd.Series(atr_values)
    atr_pct = atr_s.rolling(window, min_periods=10).rank(pct=True).fillna(0.5)
    # Invert: high ATR -> lower lookback (swings form faster in volatile markets)
    lookback = (hi - (atr_pct * (hi - lo))).round().astype(int).clip(lo, hi)
    return lookback.values

def add_swings(df, interval):
    highs = df["high"].values
    lows  = df["low"].values
    atr   = df["ict_atr_14"].values if "ict_atr_14" in df.columns else np.ones(len(df))
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
        window_h = highs[i - n: i + n + 1]
        window_l = lows[i - n: i + n + 1]
        if highs[i] == np.max(window_h):
            swing_high_flag[i]  = 1
            swing_high_price[i] = highs[i]
        if lows[i] == np.min(window_l):
            swing_low_flag[i]  = 1
            swing_low_price[i] = lows[i]

    df["ict_swing_high"]       = swing_high_flag
    df["ict_swing_low"]        = swing_low_flag
    df["ict_swing_high_price"] = swing_high_price
    df["ict_swing_low_price"]  = swing_low_price
    df["ict_swing_lookback"]   = lookbacks.astype(np.int8)
    return df

# ==============================================================================
# STEP 8: LIQUIDITY SWEEPS
# ==============================================================================

def add_liquidity_sweeps(df):
    size   = len(df)
    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values
    sh     = df["ict_swing_high"].values
    sl     = df["ict_swing_low"].values
    sh_p   = df["ict_swing_high_price"].values
    sl_p   = df["ict_swing_low_price"].values

    bull_sweep  = np.zeros(size, dtype=np.int8)
    bear_sweep  = np.zeros(size, dtype=np.int8)
    sweep_level = np.full(size, np.nan)
    intact_highs = []
    intact_lows  = []

    for i in range(size):
        if sh[i] == 1 and not np.isnan(sh_p[i]):
            intact_highs.append((i, sh_p[i]))
        if sl[i] == 1 and not np.isnan(sl_p[i]):
            intact_lows.append((i, sl_p[i]))

        # Bear sweep: wick above prior high, close back below
        swept_h = [(idx, p) for idx, p in intact_highs if highs[i] > p and closes[i] < p]
        for idx, p in swept_h:
            bear_sweep[i]  = 1
            sweep_level[i] = p
        intact_highs = [(idx, p) for idx, p in intact_highs if (idx, p) not in swept_h]

        # Bull sweep: wick below prior low, close back above
        swept_l = [(idx, p) for idx, p in intact_lows if lows[i] < p and closes[i] > p]
        for idx, p in swept_l:
            bull_sweep[i]  = 1
            sweep_level[i] = p
        intact_lows = [(idx, p) for idx, p in intact_lows if (idx, p) not in swept_l]

    df["ict_bull_liq_sweep"] = bull_sweep
    df["ict_bear_liq_sweep"] = bear_sweep
    df["ict_sweep_level"]    = sweep_level
    return df

# ==============================================================================
# STEP 9: MARKET STRUCTURE (BOS + CHoCH, close and wick versions)
# ==============================================================================

def add_market_structure(df):
    """
    Two versions:
      _close: close must exceed swing level (primary -- fewer false breaks)
      _wick:  wick must exceed swing level (sensitive -- more signals)

    ict_market_trend is the close version (primary).
    """
    size   = len(df)
    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values
    sh_p   = df["ict_swing_high_price"].values
    sl_p   = df["ict_swing_low_price"].values
    sh     = df["ict_swing_high"].values
    sl     = df["ict_swing_low"].values

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

        df[f"ict_bos_{version}"]          = bos
        df[f"ict_choch_{version}"]        = choch
        df[f"ict_market_trend_{version}"] = trend

    # Primary aliases (close version)
    df["ict_bos"]          = df["ict_bos_close"]
    df["ict_choch"]        = df["ict_choch_close"]
    df["ict_market_trend"] = df["ict_market_trend_close"]
    return df

# ==============================================================================
# STEP 10: FVG -- MULTI-TRACKING (nearest / oldest / recent + count)
# ==============================================================================

def add_fvg(df, interval="5m"):
    """
    FVG tracking with time-based lookback cap (D13 in STRATEGY_LOG).

    Root cause of previous 0.07 avg count: two competing issues
      1. Without reset: 9yr bull market means old FVGs all get tapped eventually
      2. With CHoCH reset: 5m CHoCH fires every 10-50 bars, wiping FVGs too fast
    Both approaches produced near-zero averages.

    Solution: TIME-BASED LOOKBACK CAP
    Drop any FVG older than FVG_LOOKBACK_BARS[interval] bars, regardless of
    mitigation state. This is timeframe-adaptive (~1 trading day per timeframe):
      5m: 288 bars, 15m: 96, 30m: 48, 1h: 48, 4h: 30, 1d: 20

    Mitigation: close-through rule (D12, retained)
      Bull FVG: dies when close >= fvg_top
      Bear FVG: dies when close <= fvg_bot

    In-zone flags retained (D12):
      ict_fvg_bull_in_zone: 1 when price bar overlaps any active bull FVG
      ict_fvg_bear_in_zone: 1 when price bar overlaps any active bear FVG

    Formation: standard 3-bar gap pattern
      Bull FVG: lows[i+1] > highs[i-1]
      Bear FVG: lows[i-1] > highs[i+1]
    """
    size     = len(df)
    highs    = df["high"].values
    lows     = df["low"].values
    closes   = df["close"].values
    atr      = df["ict_atr_14"].values if "ict_atr_14" in df.columns else np.ones(size)
    max_age  = FVG_LOOKBACK_BARS.get(interval, FVG_LOOKBACK_DEFAULT)

    # Formation detection
    bull_flag = np.zeros(size, dtype=np.int8)
    bear_flag = np.zeros(size, dtype=np.int8)
    bull_top_raw = np.full(size, np.nan)
    bull_bot_raw = np.full(size, np.nan)
    bear_top_raw = np.full(size, np.nan)
    bear_bot_raw = np.full(size, np.nan)

    for i in range(1, size - 1):
        gap_bull = lows[i + 1]  - highs[i - 1]
        gap_bear = lows[i - 1]  - highs[i + 1]
        min_size = atr[i] * FVG_MIN_ATR_FRACTION if not np.isnan(atr[i]) else 0
        if gap_bull > min_size:
            bull_flag[i]    = 1
            bull_top_raw[i] = lows[i + 1]
            bull_bot_raw[i] = highs[i - 1]
        if gap_bear > min_size:
            bear_flag[i]    = 1
            bear_top_raw[i] = lows[i - 1]
            bear_bot_raw[i] = highs[i + 1]

    df["ict_fvg_bull"] = bull_flag
    df["ict_fvg_bear"] = bear_flag

    # Output arrays
    out = {
        "ict_fvg_bull_recent_top":  np.full(size, np.nan),
        "ict_fvg_bull_recent_bot":  np.full(size, np.nan),
        "ict_fvg_bull_recent_mid":  np.full(size, np.nan),
        "ict_fvg_bull_recent_age":  np.full(size, np.nan),
        "ict_fvg_bull_nearest_top": np.full(size, np.nan),
        "ict_fvg_bull_nearest_bot": np.full(size, np.nan),
        "ict_fvg_bull_oldest_top":  np.full(size, np.nan),
        "ict_fvg_bull_oldest_bot":  np.full(size, np.nan),
        "ict_fvg_bull_count_above": np.zeros(size, dtype=np.int16),
        "ict_fvg_bull_count_below": np.zeros(size, dtype=np.int16),
        "ict_fvg_bull_count_total": np.zeros(size, dtype=np.int16),
        "ict_fvg_bull_in_zone":     np.zeros(size, dtype=np.int8),

        "ict_fvg_bear_recent_top":  np.full(size, np.nan),
        "ict_fvg_bear_recent_bot":  np.full(size, np.nan),
        "ict_fvg_bear_recent_mid":  np.full(size, np.nan),
        "ict_fvg_bear_recent_age":  np.full(size, np.nan),
        "ict_fvg_bear_nearest_top": np.full(size, np.nan),
        "ict_fvg_bear_nearest_bot": np.full(size, np.nan),
        "ict_fvg_bear_oldest_top":  np.full(size, np.nan),
        "ict_fvg_bear_oldest_bot":  np.full(size, np.nan),
        "ict_fvg_bear_count_above": np.zeros(size, dtype=np.int16),
        "ict_fvg_bear_count_below": np.zeros(size, dtype=np.int16),
        "ict_fvg_bear_count_total": np.zeros(size, dtype=np.int16),
        "ict_fvg_bear_in_zone":     np.zeros(size, dtype=np.int8),
    }

    active_bull = []  # list of dicts: {top, bot, mid, age, formed}
    active_bear = []

    for i in range(size):
        # AGE-BASED EXPIRY: drop FVGs older than lookback cap (D13)
        active_bull = [f for f in active_bull if f["age"] < max_age]
        active_bear = [f for f in active_bear if f["age"] < max_age]

        # MITIGATION: gap-return rule (D14)
        # Bull FVG: price RETURNS down to fill it. Mitigated when close <= fvg_top
        # (price came back down into the imbalance zone from above)
        active_bull = [f for f in active_bull if not (closes[i] <= f["top"])]
        # Bear FVG: price RETURNS up to fill it. Mitigated when close >= fvg_bot
        # (price came back up into the imbalance zone from below)
        active_bear = [f for f in active_bear if not (closes[i] >= f["bot"])]

        # Register new FVGs (AFTER mitigation/expiry checks)
        if bull_flag[i] == 1:
            mid = (bull_top_raw[i] + bull_bot_raw[i]) / 2
            active_bull.append({"top": bull_top_raw[i], "bot": bull_bot_raw[i],
                                 "mid": mid, "age": 0, "formed": i})
        if bear_flag[i] == 1:
            mid = (bear_top_raw[i] + bear_bot_raw[i]) / 2
            active_bear.append({"top": bear_top_raw[i], "bot": bear_bot_raw[i],
                                 "mid": mid, "age": 0, "formed": i})

        for f in active_bull:
            f["age"] += 1
        for f in active_bear:
            f["age"] += 1

        c = closes[i]
        h = highs[i]
        l = lows[i]

        # Bull FVG stats
        if active_bull:
            recent  = active_bull[-1]
            oldest  = active_bull[0]
            above   = [f for f in active_bull if f["bot"] > c]
            below   = [f for f in active_bull if f["top"] < c]
            nearest = min(active_bull,
                          key=lambda f: min(abs(f["top"] - c), abs(f["bot"] - c)))
            in_zone = any(l <= f["top"] and h >= f["bot"] for f in active_bull)

            out["ict_fvg_bull_recent_top"][i]  = recent["top"]
            out["ict_fvg_bull_recent_bot"][i]  = recent["bot"]
            out["ict_fvg_bull_recent_mid"][i]  = recent["mid"]
            out["ict_fvg_bull_recent_age"][i]  = recent["age"]
            out["ict_fvg_bull_nearest_top"][i] = nearest["top"]
            out["ict_fvg_bull_nearest_bot"][i] = nearest["bot"]
            out["ict_fvg_bull_oldest_top"][i]  = oldest["top"]
            out["ict_fvg_bull_oldest_bot"][i]  = oldest["bot"]
            out["ict_fvg_bull_count_above"][i] = len(above)
            out["ict_fvg_bull_count_below"][i] = len(below)
            out["ict_fvg_bull_count_total"][i] = len(active_bull)
            out["ict_fvg_bull_in_zone"][i]     = int(in_zone)

        # Bear FVG stats
        if active_bear:
            recent  = active_bear[-1]
            oldest  = active_bear[0]
            above   = [f for f in active_bear if f["bot"] > c]
            below   = [f for f in active_bear if f["top"] < c]
            nearest = min(active_bear,
                          key=lambda f: min(abs(f["top"] - c), abs(f["bot"] - c)))
            in_zone = any(l <= f["top"] and h >= f["bot"] for f in active_bear)

            out["ict_fvg_bear_recent_top"][i]  = recent["top"]
            out["ict_fvg_bear_recent_bot"][i]  = recent["bot"]
            out["ict_fvg_bear_recent_mid"][i]  = recent["mid"]
            out["ict_fvg_bear_recent_age"][i]  = recent["age"]
            out["ict_fvg_bear_nearest_top"][i] = nearest["top"]
            out["ict_fvg_bear_nearest_bot"][i] = nearest["bot"]
            out["ict_fvg_bear_oldest_top"][i]  = oldest["top"]
            out["ict_fvg_bear_oldest_bot"][i]  = oldest["bot"]
            out["ict_fvg_bear_count_above"][i] = len(above)
            out["ict_fvg_bear_count_below"][i] = len(below)
            out["ict_fvg_bear_count_total"][i] = len(active_bear)
            out["ict_fvg_bear_in_zone"][i]     = int(in_zone)

    for col, arr in out.items():
        df[col] = arr

    return df

# ==============================================================================
# STEP 11: ORDER BLOCKS -- STRICT (one OB per BOS, D06)
# ==============================================================================

def add_order_blocks(df):
    """
    Strict ICT OB definition (D06 in STRATEGY_LOG):
      Bull OB = last bearish candle before a bullish BOS (close version)
      Bear OB = last bullish candle before a bearish BOS (close version)

    One OB per BOS event. Deduplicated by searching backwards from BOS
    and stopping at the FIRST (last chronological) opposing candle.
    """
    size   = len(df)
    opens  = df["open"].values
    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values
    atr    = df["ict_atr_14"].values if "ict_atr_14" in df.columns else np.ones(size)
    bos    = df["ict_bos_close"].values

    bull_ob_flag = np.zeros(size, dtype=np.int8)
    bear_ob_flag = np.zeros(size, dtype=np.int8)
    bull_ob_top  = np.full(size, np.nan)
    bull_ob_bot  = np.full(size, np.nan)
    bear_ob_top  = np.full(size, np.nan)
    bear_ob_bot  = np.full(size, np.nan)

    assigned_bull_bos = set()
    assigned_bear_bos = set()

    for i in range(size - OB_MAX_BOS_LOOKFORWARD):
        future_bos = bos[i + 1: i + OB_MAX_BOS_LOOKFORWARD + 1]
        bull_bos_pos = np.where(future_bos == 1)[0]
        bear_bos_pos = np.where(future_bos == -1)[0]

        # Bullish OB: last bearish candle before the first bullish BOS
        if len(bull_bos_pos) > 0:
            bos_abs = i + 1 + int(bull_bos_pos[0])
            if bos_abs not in assigned_bull_bos:
                for j in range(bos_abs, i, -1):
                    if j >= size:
                        continue
                    if closes[j] < opens[j] and not np.isnan(atr[j]) and atr[j] > 0:
                        bull_ob_flag[j] = 1
                        bull_ob_top[j]  = opens[j]   # open of red candle (body top)
                        bull_ob_bot[j]  = closes[j]  # close of red candle (body bot)
                        assigned_bull_bos.add(bos_abs)
                        break

        # Bearish OB: last bullish candle before the first bearish BOS
        if len(bear_bos_pos) > 0:
            bos_abs = i + 1 + int(bear_bos_pos[0])
            if bos_abs not in assigned_bear_bos:
                for j in range(bos_abs, i, -1):
                    if j >= size:
                        continue
                    if closes[j] > opens[j] and not np.isnan(atr[j]) and atr[j] > 0:
                        bear_ob_flag[j] = 1
                        bear_ob_top[j]  = closes[j]  # close of green candle (body top)
                        bear_ob_bot[j]  = opens[j]   # open of green candle (body bot)
                        assigned_bear_bos.add(bos_abs)
                        break

    df["ict_ob_bull"] = bull_ob_flag
    df["ict_ob_bear"] = bear_ob_flag

    # Forward-fill most recent active OB with mitigation tracking
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

        # OB mitigation: price trades into the OB body
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

    df["ict_ob_bull_top"]       = out_bull_top
    df["ict_ob_bull_bot"]       = out_bull_bot
    df["ict_ob_bull_mitigated"] = out_bull_mit
    df["ict_ob_bull_age"]       = out_bull_age
    df["ict_ob_bear_top"]       = out_bear_top
    df["ict_ob_bear_bot"]       = out_bear_bot
    df["ict_ob_bear_mitigated"] = out_bear_mit
    df["ict_ob_bear_age"]       = out_bear_age
    return df

# ==============================================================================
# STEP 12: DEALING RANGE (premium / discount / OTE zone)
# ==============================================================================

def add_dealing_range(df):
    size   = len(df)
    closes = df["close"].values
    atr    = df["ict_atr_14"].values if "ict_atr_14" in df.columns else np.ones(size)
    trend  = df["ict_market_trend"].values if "ict_market_trend" in df.columns else np.zeros(size)
    sh     = df["ict_swing_high"].values
    sl     = df["ict_swing_low"].values
    sh_p   = df["ict_swing_high_price"].values
    sl_p   = df["ict_swing_low_price"].values

    dr_high   = np.full(size, np.nan)
    dr_low    = np.full(size, np.nan)
    dr_eq     = np.full(size, np.nan)
    premium   = np.zeros(size, dtype=np.int8)
    discount  = np.zeros(size, dtype=np.int8)
    ote_zone  = np.zeros(size, dtype=np.int8)
    last_sh = last_sl = np.nan

    for i in range(size):
        if sh[i] == 1 and not np.isnan(sh_p[i]):
            last_sh = sh_p[i]
        if sl[i] == 1 and not np.isnan(sl_p[i]):
            last_sl = sl_p[i]
        if np.isnan(last_sh) or np.isnan(last_sl):
            continue
        rng = last_sh - last_sl
        if np.isnan(atr[i]) or rng < atr[i] * DEALING_RANGE_MIN_ATR:
            continue
        eq = last_sl + rng * 0.5
        dr_high[i] = last_sh
        dr_low[i]  = last_sl
        dr_eq[i]   = eq
        premium[i] = 1 if closes[i] > eq else 0
        discount[i] = 1 if closes[i] <= eq else 0
        if trend[i] == 1:
            # Bullish trend: OTE is 62-79% retracement of the current leg
            if (last_sh - rng * OTE_HIGH) <= closes[i] <= (last_sh - rng * OTE_LOW):
                ote_zone[i] = 1
        elif trend[i] == -1:
            if (last_sl + rng * OTE_LOW) <= closes[i] <= (last_sl + rng * OTE_HIGH):
                ote_zone[i] = 1

    df["ict_dr_high"]  = dr_high
    df["ict_dr_low"]   = dr_low
    df["ict_dr_eq"]    = dr_eq
    df["ict_premium"]  = premium
    df["ict_discount"] = discount
    df["ict_ote_zone"] = ote_zone
    return df

# ==============================================================================
# STEP 13: OPENING GAPS (NDOG, NWOG)
# ==============================================================================

def add_opening_gaps(df):
    dt = df["bar_start_ts_utc"]
    df["_date"] = dt.dt.date
    df["_week"] = dt.dt.to_period("W")

    # Daily opening gap
    daily_close = df.groupby("_date")["close"].last().reset_index()
    daily_close.columns = ["_date", "_pdc"]
    daily_close["_pdc"]  = daily_close["_pdc"].shift(1)
    df = df.merge(daily_close, on="_date", how="left")

    daily_open = df.groupby("_date")["open"].first().reset_index()
    daily_open.columns = ["_date", "_dopen"]
    df = df.merge(daily_open, on="_date", how="left")

    gap_exists = df["_dopen"] != df["_pdc"]
    df["ict_ndog_high"]   = np.where(gap_exists, np.maximum(df["_dopen"], df["_pdc"]), df["_dopen"])
    df["ict_ndog_low"]    = np.where(gap_exists, np.minimum(df["_dopen"], df["_pdc"]), df["_dopen"])
    df["ict_ndog_mid"]    = (df["ict_ndog_high"] + df["ict_ndog_low"]) / 2
    df["ict_ndog_exists"] = gap_exists.astype(np.int8)

    # Weekly opening gap
    weekly_close = df.groupby("_week")["close"].last().reset_index()
    weekly_close.columns = ["_week", "_pwc"]
    weekly_close["_pwc"] = weekly_close["_pwc"].shift(1)
    df = df.merge(weekly_close, on="_week", how="left")

    weekly_open = df.groupby("_week")["open"].first().reset_index()
    weekly_open.columns = ["_week", "_wopen"]
    df = df.merge(weekly_open, on="_week", how="left")

    wgap = df["_wopen"] != df["_pwc"]
    df["ict_nwog_high"]   = np.where(wgap, np.maximum(df["_wopen"], df["_pwc"]), df["_wopen"])
    df["ict_nwog_low"]    = np.where(wgap, np.minimum(df["_wopen"], df["_pwc"]), df["_wopen"])
    df["ict_nwog_mid"]    = (df["ict_nwog_high"] + df["ict_nwog_low"]) / 2
    df["ict_nwog_exists"] = wgap.astype(np.int8)

    # Forward-fill gap levels through each day/week
    for col in ["ict_ndog_high", "ict_ndog_low", "ict_ndog_mid", "ict_ndog_exists"]:
        df[col] = df.groupby("_date")[col].transform("ffill")
    for col in ["ict_nwog_high", "ict_nwog_low", "ict_nwog_mid", "ict_nwog_exists"]:
        df[col] = df.groupby("_week")[col].transform("ffill")

    df.drop(columns=["_date", "_week", "_pdc", "_dopen", "_pwc", "_wopen"], inplace=True)
    return df

# ==============================================================================
# STEP 14: DISTANCE TO KEY LEVELS
# ==============================================================================

def add_distance_to_levels(df):
    close = df["close"]
    levels = [
        ("liq_pdh", "pdh"), ("liq_pdl", "pdl"),
        ("liq_pwh", "pwh"), ("liq_pwl", "pwl"),
    ]
    for col, label in levels:
        if col in df.columns:
            diff = close - df[col]
            df[f"ict_dist_{label}_dollars"] = diff
            df[f"ict_dist_{label}_pct"]     = diff / df[col] * 100
    return df

# ==============================================================================
# STEP 15: LIQUIDITY LEVELS (nearest intact swing above/below)
# ==============================================================================

def add_liquidity_levels(df):
    size   = len(df)
    closes = df["close"].values
    sh     = df["ict_swing_high"].values
    sl     = df["ict_swing_low"].values
    sh_p   = df["ict_swing_high_price"].values
    sl_p   = df["ict_swing_low_price"].values

    liq_above = np.full(size, np.nan)
    liq_below = np.full(size, np.nan)
    intact_highs = []
    intact_lows  = []

    for i in range(size):
        if sh[i] == 1 and not np.isnan(sh_p[i]):
            intact_highs.append(sh_p[i])
        if sl[i] == 1 and not np.isnan(sl_p[i]):
            intact_lows.append(sl_p[i])

        above = [p for p in intact_highs if p > closes[i]]
        below = [p for p in intact_lows  if p < closes[i]]
        if above:
            liq_above[i] = min(above)
        if below:
            liq_below[i] = max(below)

        # Remove swept levels
        intact_highs = [p for p in intact_highs if p > closes[i]]
        intact_lows  = [p for p in intact_lows  if p < closes[i]]

    df["liq_nearest_above"]         = liq_above
    df["liq_nearest_below"]         = liq_below
    df["liq_dist_above_dollars"]    = liq_above - closes
    df["liq_dist_below_dollars"]    = closes - liq_below
    df["liq_dist_above_pct"]        = (liq_above - closes) / closes * 100
    df["liq_dist_below_pct"]        = (closes - liq_below) / closes * 100
    return df

# ==============================================================================
# STEP 16: VOLUME PERCENTILE
# ==============================================================================

def add_volume_percentile(df, window=VOLUME_PCT_WINDOW):
    df["ict_volume_pct"] = df["volume_base"].rolling(window).rank(pct=True) * 100
    return df

# ==============================================================================
# STEP 17: DISPLACEMENT FLAGS (directional, gap-leaving, consecutive)
# ==============================================================================

def add_displacement(df):
    if "ict_atr_14" not in df.columns:
        for col in ["ict_disp_bull", "ict_disp_bear", "ict_disp_gap",
                    "ict_disp_consec", "ict_disp_any"]:
            df[col] = 0
        return df

    atr    = df["ict_atr_14"].values
    opens  = df["open"].values
    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values
    fvg_b  = df["ict_fvg_bull"].values if "ict_fvg_bull" in df.columns else np.zeros(len(df))
    fvg_r  = df["ict_fvg_bear"].values if "ict_fvg_bear" in df.columns else np.zeros(len(df))
    size   = len(df)

    disp_bull   = np.zeros(size, dtype=np.int8)
    disp_bear   = np.zeros(size, dtype=np.int8)
    disp_gap    = np.zeros(size, dtype=np.int8)
    disp_consec = np.zeros(size, dtype=np.int8)

    for i in range(size):
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue
        rng       = highs[i] - lows[i]
        threshold = atr[i] * DISPLACEMENT_ATR_MULT
        if rng < threshold:
            continue
        is_bull   = closes[i] > opens[i]
        is_bear   = closes[i] < opens[i]
        close_pos = (closes[i] - lows[i]) / rng if rng > 0 else 0.5
        if is_bull and close_pos >= 0.75:
            disp_bull[i] = 1
        if is_bear and close_pos <= 0.25:
            disp_bear[i] = 1

    for i in range(size):
        if (disp_bull[i] or disp_bear[i]) and (fvg_b[i] or fvg_r[i]):
            disp_gap[i] = 1

    for i in range(1, size):
        if (disp_bull[i] and disp_bull[i - 1]) or (disp_bear[i] and disp_bear[i - 1]):
            disp_consec[i] = 1

    df["ict_disp_bull"]   = disp_bull
    df["ict_disp_bear"]   = disp_bear
    df["ict_disp_gap"]    = disp_gap
    df["ict_disp_consec"] = disp_consec
    df["ict_disp_any"]    = ((disp_bull + disp_bear) > 0).astype(np.int8)
    return df

# ==============================================================================
# STEP 18: CANDLES SINCE CHoCH
# ==============================================================================

def add_candles_since_choch(df):
    choch  = df["ict_choch_close"].values
    size   = len(df)
    result = np.full(size, np.nan)
    last   = -1
    for i in range(size):
        if choch[i] != 0:
            last = i
        if last >= 0:
            result[i] = i - last
    df["ict_candles_since_choch"] = result
    return df

# ==============================================================================
# STEP 19: TIME FEATURES
# ==============================================================================

def add_time_features(df):
    dt = df["bar_start_ts_utc"]
    df["ict_hour_of_day"]  = dt.dt.hour.astype(np.int8)
    df["ict_day_of_week"]  = dt.dt.dayofweek.astype(np.int8)
    df["ict_day_of_month"] = dt.dt.day.astype(np.int8)
    df["ict_hour_sin"] = np.sin(2 * np.pi * df["ict_hour_of_day"] / 24)
    df["ict_hour_cos"] = np.cos(2 * np.pi * df["ict_hour_of_day"] / 24)
    df["ict_dow_sin"]  = np.sin(2 * np.pi * df["ict_day_of_week"] / 7)
    df["ict_dow_cos"]  = np.cos(2 * np.pi * df["ict_day_of_week"] / 7)
    return df

# ==============================================================================
# STEP 20: SESSION VWAP
# ==============================================================================

def add_session_vwap(df):
    sg  = (df["sess_label"] != df["sess_label"].shift(1)).cumsum()
    df["_sg"] = sg
    tp  = (df["high"] + df["low"] + df["close"]) / 3
    df["_tpv"] = tp * df["volume_base"]
    df["sess_vwap"] = (
        df.groupby("_sg")["_tpv"].cumsum() /
        df.groupby("_sg")["volume_base"].cumsum()
    )
    df["sess_price_vs_vwap"] = np.sign(df["close"] - df["sess_vwap"]).fillna(0).astype(np.int8)
    df.drop(columns=["_sg", "_tpv"], inplace=True)
    return df

# ==============================================================================
# STEP 21: REALIZED VOL REGIME (ATR ratio -- high-priority filter per research)
# ==============================================================================

def add_vol_regime(df):
    """
    ATR regime filter (highest-signal filter per research findings):
      ict_atr_ratio: current ATR / 50-bar rolling mean ATR
      > 1.5 = high volatility regime (trending)
      < 0.7 = low volatility regime (ranging)
    """
    if "ict_atr_14" not in df.columns:
        df["ict_atr_ratio"] = np.nan
        return df
    atr_50ma = df["ict_atr_14"].rolling(50, min_periods=10).mean()
    df["ict_atr_ratio"] = np.where(atr_50ma > 0, df["ict_atr_14"] / atr_50ma, np.nan)
    return df

# ==============================================================================
# STEP 22: FUNDING RATE DERIVED FEATURES (perp only)
# ==============================================================================

def add_funding_features(df):
    """
    Additional derived funding features on top of fund_rate_period and
    fund_rate_cum_24h that already come from resample_v2.py.
    """
    if "fund_rate_period" not in df.columns or df["fund_rate_period"].isnull().all():
        df["fund_rate_zscore"] = np.nan
        return df

    fr = df["fund_rate_period"]
    fr_std  = fr.rolling(30, min_periods=5).std()
    fr_mean = fr.rolling(30, min_periods=5).mean()
    df["fund_rate_zscore"] = np.where(
        fr_std > 0, (fr - fr_mean) / fr_std, 0.0
    )
    return df

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def enrich(instrument, interval):
    sep = "=" * 60
    log.info(sep)
    log.info(f"  {SYMBOL} | {instrument} | {interval}")
    log.info(sep)

    df = load_parquet(instrument, interval)
    if df is None:
        return

    # Store interval_mins in attrs for CVD window calculation
    df.attrs["interval_mins"] = INTERVAL_MINS.get(interval, 5)

    is_perp = (instrument == "perp")

    def step(n, total, name, fn):
        log.info(f"  [{n:02d}/{total}] {name}...")
        return fn()

    total = 22
    df = step(1,  total, "ATR",                     lambda: add_atr(df))
    df = step(2,  total, "Realized vol",             lambda: add_realized_vol(df))
    df = step(3,  total, "Session labels + SB flags",lambda: add_session_labels(df))
    df = step(4,  total, "CVD (reset variants)",     lambda: add_cvd(df))
    df = step(5,  total, "Reference prices",         lambda: add_reference_prices(df, interval))
    df = step(6,  total, "Previous levels",          lambda: add_previous_levels(df))
    df = step(7,  total, "Swings (adaptive)",        lambda: add_swings(df, interval))
    df = step(8,  total, "Liquidity sweeps",         lambda: add_liquidity_sweeps(df))
    df = step(9,  total, "Market structure",         lambda: add_market_structure(df))
    df = step(10, total, "FVG (multi-track)",        lambda: add_fvg(df, interval=interval))
    df = step(11, total, "Order blocks (strict)",    lambda: add_order_blocks(df))
    df = step(12, total, "Dealing range",            lambda: add_dealing_range(df))
    df = step(13, total, "Opening gaps (NDOG/NWOG)", lambda: add_opening_gaps(df))
    df = step(14, total, "Distance to levels",       lambda: add_distance_to_levels(df))
    df = step(15, total, "Liquidity levels",         lambda: add_liquidity_levels(df))
    df = step(16, total, "Volume percentile",        lambda: add_volume_percentile(df))
    df = step(17, total, "Displacement",             lambda: add_displacement(df))
    df = step(18, total, "Candles since CHoCH",      lambda: add_candles_since_choch(df))
    df = step(19, total, "Time features",            lambda: add_time_features(df))
    df = step(20, total, "Session VWAP",             lambda: add_session_vwap(df))
    df = step(21, total, "Vol regime (ATR ratio)",   lambda: add_vol_regime(df))
    df = step(22, total, "Funding features",         lambda: add_funding_features(df))

    # Defragment the DataFrame (eliminates PerformanceWarnings from many column additions)
    df = df.copy()

    # QA summary
    n_rows    = len(df)
    n_cols    = len(df.columns)
    ob_bull   = int(df["ict_ob_bull"].sum())
    ob_bear   = int(df["ict_ob_bear"].sum())
    fvg_avg   = float(df["ict_fvg_bull_count_total"].mean())
    disp_pct  = float(df["ict_disp_any"].sum() / n_rows * 100)
    sess_dist = df["sess_label"].value_counts().to_dict()
    sb_total  = int(df["sess_sb_london"].sum() + df["sess_sb_ny_am"].sum() + df["sess_sb_ny_pm"].sum())

    log.info(f"\n  QA - {instrument} {interval}:")
    log.info(f"    Rows:              {n_rows:,}")
    log.info(f"    Columns:           {n_cols}")
    log.info(f"    OBs:               {ob_bull} bull / {ob_bear} bear")
    log.info(f"    Avg active FVGs:   {fvg_avg:.2f}")
    log.info(f"    Displacement:      {disp_pct:.2f}% of bars")
    log.info(f"    Silver Bullet bars:{sb_total:,} ({sb_total/n_rows*100:.1f}%)")
    log.info(f"    Session dist:      {sess_dist}")

    save_parquet(df, instrument, interval)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log.info("=" * 60)
    log.info(f"  BTCDataset v2 - Step 3: ICT Enrichment v4")
    log.info(f"  Input:  {INPUT_DIR}")
    log.info(f"  Output: {OUTPUT_DIR}")
    log.info(f"  DST-aware sessions + Silver Bullet: YES")
    log.info(f"  Strict OB (one per BOS):            YES")
    log.info(f"  Multi-FVG tracking:                 YES")
    log.info(f"  Both wick+close BOS:                YES")
    log.info(f"  Real CVD (taker_buy_base):          YES")
    log.info("=" * 60)

    for instrument in INSTRUMENTS:
        for interval in INTERVALS:
            try:
                enrich(instrument, interval)
            except Exception as e:
                import traceback
                log.error(f"\n  ERROR on {instrument} {interval}: {e}")
                traceback.print_exc()

    log.info("\n" + "=" * 60)
    log.info("  Step 3 complete.")
    log.info(f"  Next: run scripts/build_master_v4.py")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
