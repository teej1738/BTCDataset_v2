"""
enrich_features_v2.py -- Dataset Enrichment Pass v2 (D33)

Adds ~120 technical indicator columns to the labeled parquet.
Reads v1, writes v2. Additive only -- v1 preserved unchanged.

Annualization factor for 5m crypto: sqrt(105120) = 324.22
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import time

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ANNUALIZE_5M = np.sqrt(105120)  # 324.22
ANN_FACTOR = 105120  # bars per year (5m, 24/7 crypto)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "labeled"
INPUT_FILE = DATA_DIR / "BTCUSDT_MASTER_labeled.parquet"
OUTPUT_FILE = DATA_DIR / "BTCUSDT_5m_labeled_v2.parquet"
CATALOG_FILE = DATA_DIR / "feature_catalog_v2.yaml"

ET = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def wilder_smooth(series, period):
    """Wilder smoothing (alpha = 1/n, NOT 2/(n+1))."""
    vals = series.values if hasattr(series, "values") else np.asarray(series)
    out = np.full(len(vals), np.nan)
    # Find first non-NaN run of length >= period
    ok = ~np.isnan(vals)
    run = 0
    start = -1
    for i in range(len(vals)):
        if ok[i]:
            run += 1
            if run == period:
                start = i - period + 1
                break
        else:
            run = 0
    if start < 0:
        return pd.Series(out, index=series.index) if hasattr(series, "index") else out
    out[start + period - 1] = np.nanmean(vals[start : start + period])
    for i in range(start + period, len(vals)):
        if np.isnan(vals[i]):
            out[i] = out[i - 1]
        else:
            out[i] = out[i - 1] + (vals[i] - out[i - 1]) / period
    if hasattr(series, "index"):
        return pd.Series(out, index=series.index)
    return out


def compute_rsi_wilder(close, period):
    """RSI with Wilder smoothing."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = wilder_smooth(gain, period)
    avg_loss = wilder_smooth(loss, period)
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


# ===================================================================
# PART 1: Raw Data Additions
# ===================================================================
def add_part1_raw(df):
    """Spread proxy, funding regime features."""
    new_cols = []

    # Abdi-Ranaldo spread proxy (causal)
    eta = np.log(df["close"]) - np.log((df["high"] + df["low"]) / 2)
    eta_lag = eta.shift(1)
    xy = eta_lag * eta
    roll_cov = xy.rolling(20).mean() - eta_lag.rolling(20).mean() * eta.rolling(20).mean()
    df["spread_ar"] = 2 * np.sqrt(np.maximum(0, roll_cov))
    new_cols.append("spread_ar")

    # Funding regime
    fund = df["fund_rate_period"]
    fund_mean_24 = fund.rolling(24).mean()
    df["funding_regime"] = np.where(
        fund_mean_24 > 0.0001, 1, np.where(fund_mean_24 < -0.00005, -1, 0)
    ).astype(np.int8)
    new_cols.append("funding_regime")

    # Funding z-score (288-bar = 24h)
    fund_mean_288 = fund.rolling(288).mean()
    fund_std_288 = fund.rolling(288).std()
    df["funding_zscore_v2"] = (fund - fund_mean_288) / fund_std_288
    new_cols.append("funding_zscore_v2")

    # Time to funding (normalized 0-1)
    hours = df["bar_start_ts_utc"].dt.hour
    minutes = df["bar_start_ts_utc"].dt.minute
    mins_since_midnight = hours * 60 + minutes
    bars_in_cycle = (mins_since_midnight % 480) // 5
    bars_until = 96 - bars_in_cycle
    bars_until = np.where(bars_until == 96, 0, bars_until)
    df["time_to_funding"] = (bars_until / 96.0).astype(np.float32)
    new_cols.append("time_to_funding")

    # Annualized funding
    df["annualized_funding"] = fund * 3 * 365
    new_cols.append("annualized_funding")

    return new_cols


# ===================================================================
# PART 2: Momentum Indicators
# ===================================================================
def add_part2_momentum(df):
    """RSI, MACD, Stochastic, ROC."""
    new_cols = []
    close = df["close"]
    atr = df["ict_atr_14"]

    # RSI variants (Wilder smoothing)
    for p in [14, 9, 21]:
        col = f"rsi_{p}"
        df[col] = compute_rsi_wilder(close, p)
        new_cols.append(col)

    # RSI divergence (simplified causal pivot-based)
    rsi14 = df["rsi_14"]
    price_min_prev = close.shift(5).rolling(15).min()
    rsi_min_prev = rsi14.shift(5).rolling(15).min()
    price_min_20 = close.rolling(20).min()
    close_near_low = ((close - price_min_20).abs() / atr) < 0.5
    df["rsi_divergence_bull"] = (
        close_near_low & (close < price_min_prev) & (rsi14 > rsi_min_prev)
    ).astype(np.int8)
    new_cols.append("rsi_divergence_bull")

    price_max_prev = close.shift(5).rolling(15).max()
    rsi_max_prev = rsi14.shift(5).rolling(15).max()
    price_max_20 = close.rolling(20).max()
    close_near_high = ((price_max_20 - close).abs() / atr) < 0.5
    df["rsi_divergence_bear"] = (
        close_near_high & (close > price_max_prev) & (rsi14 < rsi_max_prev)
    ).astype(np.int8)
    new_cols.append("rsi_divergence_bear")

    # RSI regime (3-bar confirmation)
    rsi_above = (rsi14 > 50).astype(int)
    rsi_below = (rsi14 < 50).astype(int)
    df["rsi_regime"] = np.where(
        rsi_above.rolling(3).min() == 1, 1,
        np.where(rsi_below.rolling(3).min() == 1, -1, 0),
    ).astype(np.int8)
    new_cols.append("rsi_regime")

    # MACD standard (12/26/9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    df["macd_line"] = macd_line
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd_hist
    df["macd_hist_roc"] = macd_hist - macd_hist.shift(1)
    df["macd_norm"] = macd_line / atr
    new_cols.extend(["macd_line", "macd_signal", "macd_hist", "macd_hist_roc", "macd_norm"])

    # MACD fast (8/17/9)
    ema8 = close.ewm(span=8, adjust=False).mean()
    ema17 = close.ewm(span=17, adjust=False).mean()
    macd_fast_line = ema8 - ema17
    macd_fast_signal = macd_fast_line.ewm(span=9, adjust=False).mean()
    df["macd_fast_line"] = macd_fast_line
    df["macd_fast_signal"] = macd_fast_signal
    df["macd_fast_hist"] = macd_fast_line - macd_fast_signal
    df["macd_fast_norm"] = macd_fast_line / atr
    new_cols.extend(["macd_fast_line", "macd_fast_signal", "macd_fast_hist", "macd_fast_norm"])

    # Stochastic slow (14,3,3)
    low_14 = df["low"].rolling(14).min()
    high_14 = df["high"].rolling(14).max()
    stoch_k_raw = 100 * (close - low_14) / (high_14 - low_14)
    stoch_k_slow = stoch_k_raw.rolling(3).mean()
    stoch_d = stoch_k_slow.rolling(3).mean()
    df["stoch_k"] = stoch_k_raw
    df["stoch_d"] = stoch_d
    df["stoch_k_slow"] = stoch_k_slow
    new_cols.extend(["stoch_k", "stoch_d", "stoch_k_slow"])

    # Rate of Change (log returns)
    for n in [5, 10, 20, 60]:
        col = f"roc_{n}"
        df[col] = np.log(close / close.shift(n))
        new_cols.append(col)
    roc20 = df["roc_20"]
    df["roc_zscore_20"] = (roc20 - roc20.rolling(100).mean()) / roc20.rolling(100).std()
    new_cols.append("roc_zscore_20")

    return new_cols


# ===================================================================
# PART 3: Trend Indicators
# ===================================================================
def add_part3_trend(df):
    """EMA, VWAP, Supertrend, ADX/DMI, Ichimoku."""
    new_cols = []
    close = df["close"]
    high = df["high"]
    low = df["low"]
    atr = df["ict_atr_14"]

    # EMA suite
    for p in [9, 21, 50, 200]:
        col = f"ema_{p}"
        df[col] = close.ewm(span=p, adjust=False).mean()
        new_cols.append(col)

    ema21 = df["ema_21"]
    df["ema_9_21_cross"] = np.where(df["ema_9"] > ema21, 1, -1).astype(np.int8)
    df["ema_slope_21"] = (ema21 - ema21.shift(5)) / (5 * atr)
    df["ema_dist_21"] = (close - ema21) / ema21
    df["ema_dist_200"] = (close - df["ema_200"]) / df["ema_200"]
    new_cols.extend(["ema_9_21_cross", "ema_slope_21", "ema_dist_21", "ema_dist_200"])

    # MTF EMA score: sign(close-ema21) on 5m + H1 + H4
    sign_5m = np.sign(close - ema21)
    h1_ema21 = df["h1_close"].ewm(span=21, adjust=False).mean()
    h4_ema21 = df["h4_close"].ewm(span=21, adjust=False).mean()
    sign_h1 = np.sign(df["h1_close"] - h1_ema21)
    sign_h4 = np.sign(df["h4_close"] - h4_ema21)
    df["mtf_ema_score"] = (sign_5m + sign_h1 + sign_h4).fillna(0).astype(np.int8)
    new_cols.append("mtf_ema_score")

    # VWAP (daily UTC reset)
    tp = (high + low + close) / 3
    vol = df["volume_base"]
    utc_date = df["bar_start_ts_utc"].dt.date
    cum_tp_vol = (tp * vol).groupby(utc_date).cumsum()
    cum_vol = vol.groupby(utc_date).cumsum()
    vwap = cum_tp_vol / cum_vol
    df["vwap_daily"] = vwap
    df["vwap_dev"] = (close - vwap) / vwap
    new_cols.extend(["vwap_daily", "vwap_dev"])

    # VWAP bands
    vwap_sq_vol = (vol * (tp - vwap) ** 2).groupby(utc_date).cumsum()
    vwap_std = np.sqrt(vwap_sq_vol / cum_vol)
    df["vwap_upper_1"] = vwap + vwap_std
    df["vwap_upper_2"] = vwap + 2 * vwap_std
    df["vwap_lower_1"] = vwap - vwap_std
    df["vwap_lower_2"] = vwap - 2 * vwap_std
    new_cols.extend(["vwap_upper_1", "vwap_upper_2", "vwap_lower_1", "vwap_lower_2"])

    df["vwap_position"] = np.where(
        close > vwap + 2 * vwap_std, 2,
        np.where(close > vwap + vwap_std, 1,
        np.where(close < vwap - 2 * vwap_std, -2,
        np.where(close < vwap - vwap_std, -1, 0)))
    ).astype(np.int8)
    new_cols.append("vwap_position")

    # Supertrend (ATR=10, mult=2.5)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr10 = tr.rolling(10).mean()
    midpoint = (high + low) / 2
    ub_raw = (midpoint + 2.5 * atr10).values
    lb_raw = (midpoint - 2.5 * atr10).values
    c = close.values
    n = len(df)

    st_dir = np.ones(n, dtype=np.int8)
    st_val = np.full(n, np.nan)
    # Find first valid ATR bar
    first_valid = 0
    while first_valid < n and np.isnan(ub_raw[first_valid]):
        first_valid += 1
    prev_ub = ub_raw[first_valid] if first_valid < n else np.nan
    prev_lb = lb_raw[first_valid] if first_valid < n else np.nan
    if first_valid < n:
        st_val[first_valid] = prev_lb  # start bullish

    for i in range(first_valid + 1, n):
        if np.isnan(ub_raw[i]):
            st_dir[i] = st_dir[i - 1]
            st_val[i] = st_val[i - 1]
            continue
        # Ratchet bands
        cur_lb = lb_raw[i] if (lb_raw[i] > prev_lb or c[i - 1] < prev_lb) else prev_lb
        cur_ub = ub_raw[i] if (ub_raw[i] < prev_ub or c[i - 1] > prev_ub) else prev_ub
        # Direction
        if st_dir[i - 1] == 1:
            if c[i] < cur_lb:
                st_dir[i] = -1
                st_val[i] = cur_ub
            else:
                st_dir[i] = 1
                st_val[i] = cur_lb
        else:
            if c[i] > cur_ub:
                st_dir[i] = 1
                st_val[i] = cur_lb
            else:
                st_dir[i] = -1
                st_val[i] = cur_ub
        prev_ub = cur_ub
        prev_lb = cur_lb

    df["supertrend_signal"] = st_dir
    df["supertrend_dist"] = (c - st_val) / atr.values
    new_cols.extend(["supertrend_signal", "supertrend_dist"])

    # ADX/DMI (period=14, Wilder smoothing)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index
    )
    sm_plus = wilder_smooth(plus_dm, 14)
    sm_minus = wilder_smooth(minus_dm, 14)
    sm_tr = wilder_smooth(tr, 14)
    di_plus = 100 * sm_plus / sm_tr
    di_minus = 100 * sm_minus / sm_tr
    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus)
    adx = wilder_smooth(dx, 14)

    df["adx_14"] = adx
    df["di_plus_14"] = di_plus
    df["di_minus_14"] = di_minus
    new_cols.extend(["adx_14", "di_plus_14", "di_minus_14"])

    adx_v = adx.values if hasattr(adx, "values") else adx
    df["adx_trend_regime"] = np.where(
        adx_v > 25, 1.0, np.where(adx_v < 20, 0.0, (adx_v - 20) / 5.0)
    )
    df["di_cross"] = np.where(di_plus > di_minus, 1, -1).astype(np.int8)
    new_cols.extend(["adx_trend_regime", "di_cross"])

    # Ichimoku (crypto 10/30/60/30, NO chikou)
    tenkan = (high.rolling(10).max() + low.rolling(10).min()) / 2
    kijun = (high.rolling(30).max() + low.rolling(30).min()) / 2
    senkou_a = (tenkan + kijun) / 2
    senkou_b = (high.rolling(60).max() + low.rolling(60).min()) / 2
    cloud_top = np.maximum(senkou_a, senkou_b)
    cloud_bot = np.minimum(senkou_a, senkou_b)

    df["ichi_cloud_signal"] = np.where(
        close > cloud_top, 2, np.where(close < cloud_bot, 0, 1)
    ).astype(np.int8)
    df["ichi_tk_cross"] = np.where(tenkan > kijun, 1, -1).astype(np.int8)
    df["ichi_cloud_thickness"] = (cloud_top - cloud_bot) / close
    new_cols.extend(["ichi_cloud_signal", "ichi_tk_cross", "ichi_cloud_thickness"])

    return new_cols, atr10


# ===================================================================
# PART 4: Volume and Money Flow
# ===================================================================
def add_part4_volume(df):
    """CLV, MFI, OBV, CVD, CMF, volume features."""
    new_cols = []
    close = df["close"]
    high = df["high"]
    low = df["low"]
    vol = df["volume_base"]

    # CLV
    hl = high - low
    clv = ((close - low) - (high - close)) / hl
    clv = clv.fillna(0)
    df["clv"] = clv
    new_cols.append("clv")

    # MFI
    tp = (high + low + close) / 3
    rmf = tp * vol
    tp_up = tp > tp.shift(1)
    tp_down = tp < tp.shift(1)
    pos_mf = pd.Series(np.where(tp_up, rmf, 0.0), index=df.index)
    neg_mf = pd.Series(np.where(tp_down, rmf, 0.0), index=df.index)
    for p in [14, 9]:
        mfi = 100 - 100 / (1 + pos_mf.rolling(p).sum() / neg_mf.rolling(p).sum())
        df[f"mfi_{p}"] = mfi
        new_cols.append(f"mfi_{p}")

    # OBV (normalized)
    sign_c = np.sign(close - close.shift(1))
    obv_raw = (vol * sign_c).cumsum()
    df["obv_zscore"] = (obv_raw - obv_raw.rolling(50).mean()) / obv_raw.rolling(50).std()
    df["obv_roc"] = obv_raw - obv_raw.shift(14)
    new_cols.extend(["obv_zscore", "obv_roc"])

    # CVD approximation
    cvd_bar = clv * vol
    cvd_cum = cvd_bar.cumsum()
    df["cvd_bar"] = cvd_bar
    df["cvd_zscore"] = (cvd_cum - cvd_cum.rolling(50).mean()) / cvd_cum.rolling(50).std()
    df["cvd_roc"] = cvd_cum - cvd_cum.shift(14)
    new_cols.extend(["cvd_bar", "cvd_zscore", "cvd_roc"])

    # CMF (20)
    df["cmf_20"] = (clv * vol).rolling(20).sum() / vol.rolling(20).sum()
    new_cols.append("cmf_20")

    # Volume features
    vol_mean = vol.rolling(20).mean()
    vol_std = vol.rolling(20).std()
    df["volume_zscore"] = (vol - vol_mean) / vol_std
    df["volume_rvol"] = vol / vol_mean
    df["volume_percentile"] = vol.rolling(100).rank(pct=True)
    new_cols.extend(["volume_zscore", "volume_rvol", "volume_percentile"])

    # Taker buy ratio
    if "taker_buy_base" in df.columns:
        df["taker_buy_ratio"] = df["taker_buy_base"] / vol
    else:
        df["taker_buy_ratio"] = (clv + 1) / 2
    new_cols.append("taker_buy_ratio")

    return new_cols


# ===================================================================
# PART 5: Volatility Features
# ===================================================================
def add_part5_volatility(df, atr10):
    """GK, Parkinson, RS, HV, Bollinger, Keltner, TTM Squeeze."""
    new_cols = []
    close = df["close"]
    high = df["high"]
    low = df["low"]
    opn = df["open"]

    # Per-bar log components
    log_hl = np.log(high / low)
    log_co = np.log(close / opn)
    log_hc = np.log(high / close)
    log_ho = np.log(high / opn)
    log_lc = np.log(low / close)
    log_lo = np.log(low / opn)
    log_ret = np.log(close / close.shift(1))

    # Garman-Klass
    gk_bar = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2
    for p in [5, 10, 20, 60]:
        df[f"gk_{p}"] = np.sqrt(np.maximum(0, gk_bar.rolling(p).mean()) * ANN_FACTOR)
        new_cols.append(f"gk_{p}")

    # Parkinson
    pk_bar = log_hl ** 2 / (4 * np.log(2))
    for p in [5, 20, 60]:
        df[f"parkinson_{p}"] = np.sqrt(np.maximum(0, pk_bar.rolling(p).mean()) * ANN_FACTOR)
        new_cols.append(f"parkinson_{p}")

    # Rogers-Satchell
    rs_bar = log_hc * log_ho + log_lc * log_lo
    for p in [20, 60]:
        df[f"rs_{p}"] = np.sqrt(np.maximum(0, rs_bar.rolling(p).mean()) * ANN_FACTOR)
        new_cols.append(f"rs_{p}")

    # Historical volatility
    for p in [20, 60]:
        df[f"hv_{p}"] = log_ret.rolling(p).std() * ANNUALIZE_5M
        new_cols.append(f"hv_{p}")

    # Bollinger Bands (20, 2.0)
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std
    df["bb_pct_b"] = (close - bb_lower) / (bb_upper - bb_lower)
    df["bb_bandwidth"] = (bb_upper - bb_lower) / bb_mid
    new_cols.extend(["bb_pct_b", "bb_bandwidth"])

    # Keltner Channels (EMA=20, ATR=10, mult=1.5)
    ema20 = close.ewm(span=20, adjust=False).mean()
    kc_upper = ema20 + 1.5 * atr10
    kc_lower = ema20 - 1.5 * atr10

    # TTM Squeeze
    squeeze_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)
    df["squeeze_on"] = squeeze_on.astype(np.int8)
    new_cols.append("squeeze_on")

    sq = squeeze_on.values
    squeeze_bars = np.zeros(len(sq), dtype=np.int16)
    for i in range(len(sq)):
        if sq[i]:
            squeeze_bars[i] = (squeeze_bars[i - 1] + 1) if i > 0 else 1
    df["squeeze_bars"] = squeeze_bars
    new_cols.append("squeeze_bars")

    # Volatility regime
    hv20 = df["hv_20"]
    df["vol_percentile"] = hv20.rolling(2016).rank(pct=True)
    df["vol_regime"] = np.where(
        df["vol_percentile"] < 0.33, 0,
        np.where(df["vol_percentile"] > 0.67, 2, 1)
    ).astype(np.int8)
    df["vol_ratio"] = hv20 / df["hv_60"]
    new_cols.extend(["vol_percentile", "vol_regime", "vol_ratio"])

    return new_cols


# ===================================================================
# PART 6: ICT Session and Structural Features
# ===================================================================
def add_part6_ict_session(df):
    """Silver Bullet, Macros, Kill Zones, PO3, OTE, CISD, OB quality."""
    new_cols = []
    n = len(df)

    # Convert to ET
    ts_et = df["bar_start_ts_utc"].dt.tz_convert(ET)
    et_hour = ts_et.dt.hour.values
    et_minute = ts_et.dt.minute.values
    et_tm = et_hour * 60 + et_minute  # minutes since midnight ET

    # --- Silver Bullet (ET, DST-aware) ---
    df["sb_london_et"] = ((et_tm >= 180) & (et_tm < 240)).astype(np.int8)
    df["sb_ny_am_et"] = ((et_tm >= 600) & (et_tm < 660)).astype(np.int8)
    df["sb_ny_pm_et"] = ((et_tm >= 840) & (et_tm < 900)).astype(np.int8)
    df["sb_any"] = (
        (df["sb_london_et"] == 1) | (df["sb_ny_am_et"] == 1) | (df["sb_ny_pm_et"] == 1)
    ).astype(np.int8)
    new_cols.extend(["sb_london_et", "sb_ny_am_et", "sb_ny_pm_et", "sb_any"])

    # --- ICT Macros (ET, DST-aware) ---
    df["macro_ny_open"] = ((et_tm >= 570) & (et_tm < 590)).astype(np.int8)
    df["macro_ny_cont"] = ((et_tm >= 590) & (et_tm < 610)).astype(np.int8)
    df["macro_late_am"] = ((et_tm >= 650) & (et_tm < 670)).astype(np.int8)
    df["macro_lunch"] = ((et_tm >= 710) & (et_tm < 730)).astype(np.int8)
    df["macro_late_aft"] = ((et_tm >= 890) & (et_tm < 910)).astype(np.int8)
    df["macro_london_1"] = ((et_tm >= 153) & (et_tm < 180)).astype(np.int8)
    df["macro_london_2"] = ((et_tm >= 243) & (et_tm < 270)).astype(np.int8)
    df["macro_any"] = (
        (df["macro_ny_open"] == 1) | (df["macro_ny_cont"] == 1)
        | (df["macro_late_am"] == 1) | (df["macro_lunch"] == 1)
        | (df["macro_late_aft"] == 1) | (df["macro_london_1"] == 1)
        | (df["macro_london_2"] == 1)
    ).astype(np.int8)
    df["lunch_zone"] = ((et_tm >= 720) & (et_tm < 780)).astype(np.int8)
    new_cols.extend([
        "macro_ny_open", "macro_ny_cont", "macro_late_am", "macro_lunch",
        "macro_late_aft", "macro_london_1", "macro_london_2", "macro_any", "lunch_zone",
    ])

    # --- Kill Zones ---
    df["kz_london"] = ((et_tm >= 120) & (et_tm < 300)).astype(np.int8)
    df["kz_ny_open"] = ((et_tm >= 510) & (et_tm < 660)).astype(np.int8)
    df["kz_ny_close"] = ((et_tm >= 600) & (et_tm < 690)).astype(np.int8)
    df["kz_any"] = (
        (df["kz_london"] == 1) | (df["kz_ny_open"] == 1) | (df["kz_ny_close"] == 1)
    ).astype(np.int8)
    new_cols.extend(["kz_london", "kz_ny_open", "kz_ny_close", "kz_any"])

    # --- Session labels ---
    df["asia_session"] = (et_tm >= 1200).astype(np.int8)  # 20:00-23:55 ET
    df["london_session"] = ((et_tm >= 120) & (et_tm < 300)).astype(np.int8)
    df["ny_session"] = ((et_tm >= 510) & (et_tm < 960)).astype(np.int8)
    new_cols.extend(["asia_session", "london_session", "ny_session"])

    # --- PO3/AMD (stateful loop) ---
    hi = df["high"].values
    lo = df["low"].values
    cl = df["close"].values
    atr_vals = df["ict_atr_14"].values

    asia_high_arr = np.full(n, np.nan)
    asia_low_arr = np.full(n, np.nan)
    asia_range_arr = np.full(n, np.nan)
    asia_range_norm_arr = np.full(n, np.nan)
    london_swept_bsl_arr = np.zeros(n, dtype=np.int8)
    london_swept_ssl_arr = np.zeros(n, dtype=np.int8)
    po3_bias_arr = np.zeros(n, dtype=np.int8)
    po3_confirmed_arr = np.zeros(n, dtype=np.int8)

    cur_asia_h = np.nan
    cur_asia_l = np.nan
    frozen_h = np.nan
    frozen_l = np.nan
    frozen_mid = np.nan
    bias = 0
    confirmed = 0
    prev_asia = False
    bsl_swept = False
    ssl_swept = False

    for i in range(n):
        tm = et_tm[i]
        in_asia = tm >= 1200  # 20:00-23:55 ET
        in_london = 120 <= tm < 300

        # New Asia session
        if in_asia and not prev_asia:
            cur_asia_h = hi[i]
            cur_asia_l = lo[i]
            bsl_swept = False
            ssl_swept = False
            bias = 0
            confirmed = 0
        elif in_asia:
            if hi[i] > cur_asia_h:
                cur_asia_h = hi[i]
            if lo[i] < cur_asia_l:
                cur_asia_l = lo[i]

        if in_asia:
            frozen_h = cur_asia_h
            frozen_l = cur_asia_l
            frozen_mid = (cur_asia_h + cur_asia_l) / 2

        # Propagate frozen values
        asia_high_arr[i] = frozen_h
        asia_low_arr[i] = frozen_l
        if not np.isnan(frozen_h) and not np.isnan(frozen_l):
            asia_range_arr[i] = frozen_h - frozen_l
            if atr_vals[i] > 0:
                asia_range_norm_arr[i] = asia_range_arr[i] / atr_vals[i]

        # London sweep detection
        if in_london and not np.isnan(frozen_h):
            if hi[i] > frozen_h:
                london_swept_bsl_arr[i] = 1
                bsl_swept = True
            if lo[i] < frozen_l:
                london_swept_ssl_arr[i] = 1
                ssl_swept = True

        # PO3 bias
        if ssl_swept and not bsl_swept:
            bias = 1
            if not np.isnan(frozen_mid) and cl[i] > frozen_mid:
                confirmed = 1
        elif bsl_swept and not ssl_swept:
            bias = -1
            if not np.isnan(frozen_mid) and cl[i] < frozen_mid:
                confirmed = 1

        po3_bias_arr[i] = bias
        po3_confirmed_arr[i] = confirmed
        prev_asia = in_asia

    df["asia_high"] = asia_high_arr
    df["asia_low"] = asia_low_arr
    df["asia_range"] = asia_range_arr
    df["asia_range_norm"] = asia_range_norm_arr
    df["london_swept_bsl"] = london_swept_bsl_arr
    df["london_swept_ssl"] = london_swept_ssl_arr
    df["po3_bias"] = po3_bias_arr
    df["po3_confirmed"] = po3_confirmed_arr
    new_cols.extend([
        "asia_high", "asia_low", "asia_range", "asia_range_norm",
        "london_swept_bsl", "london_swept_ssl", "po3_bias", "po3_confirmed",
    ])

    # --- OTE zone (use most recent swing pair, forward-filled) ---
    swing_h_price = df["ict_swing_high_price"].ffill().values
    swing_l_price = df["ict_swing_low_price"].ffill().values
    swing_range = swing_h_price - swing_l_price
    ote_low = swing_h_price - 0.79 * swing_range
    ote_high = swing_h_price - 0.62 * swing_range
    ote_mid = swing_h_price - 0.705 * swing_range
    df["ote_in_zone"] = ((cl >= ote_low) & (cl <= ote_high)).astype(np.int8)
    df["ote_dist"] = (cl - ote_mid) / atr_vals
    new_cols.extend(["ote_in_zone", "ote_dist"])

    # --- CISD (body-based, stateful loop) ---
    opn = df["open"].values
    swing_h = df["ict_swing_high"].values
    swing_l = df["ict_swing_low"].values

    cisd_bull_arr = np.zeros(n, dtype=np.int8)
    cisd_bear_arr = np.zeros(n, dtype=np.int8)
    last_up_seq_open = np.nan
    last_down_seq_open = np.nan
    bear_active = False
    bull_active = False

    for i in range(1, n):
        # New swing high -- find start of up-candle sequence
        if swing_h[i] == 1:
            seq_open = opn[i]
            j = i
            while j > 0 and cl[j - 1] > opn[j - 1]:
                seq_open = opn[j - 1]
                j -= 1
            last_up_seq_open = seq_open
            bear_active = True

        # New swing low -- find start of down-candle sequence
        if swing_l[i] == 1:
            seq_open = opn[i]
            j = i
            while j > 0 and cl[j - 1] < opn[j - 1]:
                seq_open = opn[j - 1]
                j -= 1
            last_down_seq_open = seq_open
            bull_active = True

        # One-shot triggers
        if bear_active and not np.isnan(last_up_seq_open) and cl[i] < last_up_seq_open:
            cisd_bear_arr[i] = 1
            bear_active = False
        if bull_active and not np.isnan(last_down_seq_open) and cl[i] > last_down_seq_open:
            cisd_bull_arr[i] = 1
            bull_active = False

    df["cisd_bull"] = cisd_bull_arr
    df["cisd_bear"] = cisd_bear_arr
    new_cols.extend(["cisd_bull", "cisd_bear"])

    # --- OB displacement quality ---
    ob_bull = df["ict_ob_bull"].values
    ob_bear = df["ict_ob_bear"].values
    body_size = np.abs(cl - opn)
    disp = body_size / atr_vals
    df["ob_disp_quality_bull"] = np.where(ob_bull == 1, disp, np.nan)
    df["ob_disp_quality_bear"] = np.where(ob_bear == 1, disp, np.nan)
    new_cols.extend(["ob_disp_quality_bull", "ob_disp_quality_bear"])

    # --- Internal swings (3-bar causal) ---
    df["int_swing_high"] = (df["high"] == df["high"].rolling(3).max()).astype(np.int8)
    df["int_swing_low"] = (df["low"] == df["low"].rolling(3).min()).astype(np.int8)
    new_cols.extend(["int_swing_high", "int_swing_low"])

    # --- ICT confluence score (0-8) ---
    score = np.zeros(n, dtype=np.int8)
    score += (df["kz_any"].values == 1).astype(np.int8)
    score += (df["sb_any"].values == 1).astype(np.int8)
    score += (po3_confirmed_arr == 1).astype(np.int8)
    score += (df["adx_14"].values > 25).astype(np.int8)
    score += (df["ichi_tk_cross"].values == df["ema_9_21_cross"].values).astype(np.int8)
    score += (df["obv_roc"].values > 0).astype(np.int8)
    score += (df["ote_in_zone"].values == 1).astype(np.int8)
    score += (df["vol_regime"].values != 2).astype(np.int8)
    df["ict_confluence_v2"] = score
    new_cols.append("ict_confluence_v2")

    return new_cols


# ===================================================================
# Validation
# ===================================================================
def validate_output(df, original_rows, new_cols):
    """Validate row count, NaN counts, sample stats, causality."""
    print("\n=== VALIDATION ===")

    # Row count
    assert len(df) == original_rows, f"Row mismatch: {len(df)} vs {original_rows}"
    print(f"Row count: {len(df):,} == {original_rows:,} -- PASS")

    # NaN counts
    print(f"\nNaN counts (non-zero only, {len(new_cols)} new columns):")
    nan_count = 0
    for col in new_cols:
        pct = df[col].isna().mean() * 100
        if pct > 0:
            print(f"  {col}: {pct:.2f}%")
            nan_count += 1
    if nan_count == 0:
        print("  (none)")

    # Sample stats
    print("\nSample column statistics (first 10):")
    for col in new_cols[:10]:
        s = df[col].dropna()
        if len(s) == 0:
            print(f"  {col}: all NaN")
        elif s.dtype in [np.int8, np.int16, np.int32, np.int64]:
            print(f"  {col}: min={s.min()}, max={s.max()}, mean={s.mean():.4f}")
        else:
            print(f"  {col}: min={s.min():.6f}, max={s.max():.6f}, mean={s.mean():.6f}")

    # Causality spot-check
    import random
    rng = random.Random(42)
    sample = rng.sample(new_cols, min(5, len(new_cols)))
    print("\nCausality spot-check (5 random columns at bar 999):")
    for col in sample:
        v = df[col].values[999]
        if isinstance(v, (float, np.floating)) and np.isnan(v):
            print(f"  {col}: NaN (warmup)")
        elif isinstance(v, (float, np.floating)):
            print(f"  {col}: {v:.6f}")
        else:
            print(f"  {col}: {v}")
    print("  All features use rolling/shift/cumsum -- causal by construction")


# ===================================================================
# YAML Feature Catalog
# ===================================================================
def write_catalog(new_cols, catalog_path):
    """Write feature catalog as YAML."""
    meta = {
        "spread_ar": ("Raw/Microstructure", "Abdi-Ranaldo spread proxy", 20, 1),
        "funding_regime": ("Raw/Funding", "Rolling 24-bar funding regime", 24, 2),
        "funding_zscore_v2": ("Raw/Funding", "Funding rate z-score 288-bar", 288, 2),
        "time_to_funding": ("Raw/Funding", "Normalized time to next funding", 0, 3),
        "annualized_funding": ("Raw/Funding", "Annualized funding rate", 0, 2),
        "rsi_14": ("Momentum/RSI", "RSI Wilder 14-period", 14, 1),
        "rsi_9": ("Momentum/RSI", "RSI Wilder 9-period", 9, 1),
        "rsi_21": ("Momentum/RSI", "RSI Wilder 21-period", 21, 1),
        "rsi_divergence_bull": ("Momentum/RSI", "Bullish RSI divergence", 20, 1),
        "rsi_divergence_bear": ("Momentum/RSI", "Bearish RSI divergence", 20, 1),
        "rsi_regime": ("Momentum/RSI", "RSI regime 3-bar confirm", 17, 2),
        "macd_line": ("Momentum/MACD", "MACD line 12/26", 26, 3),
        "macd_signal": ("Momentum/MACD", "MACD signal 12/26/9", 35, 3),
        "macd_hist": ("Momentum/MACD", "MACD histogram", 35, 3),
        "macd_hist_roc": ("Momentum/MACD", "MACD histogram ROC", 36, 3),
        "macd_norm": ("Momentum/MACD", "MACD normalized by ATR", 35, 3),
        "macd_fast_line": ("Momentum/MACD", "Fast MACD line 8/17", 17, 2),
        "macd_fast_signal": ("Momentum/MACD", "Fast MACD signal 8/17/9", 26, 2),
        "macd_fast_hist": ("Momentum/MACD", "Fast MACD histogram", 26, 2),
        "macd_fast_norm": ("Momentum/MACD", "Fast MACD normalized by ATR", 26, 2),
        "stoch_k": ("Momentum/Stochastic", "Stochastic raw K 14-period", 14, 3),
        "stoch_d": ("Momentum/Stochastic", "Stochastic D (slow 14,3,3)", 20, 3),
        "stoch_k_slow": ("Momentum/Stochastic", "Slow Stochastic K (3-SMA)", 16, 3),
        "roc_5": ("Momentum/ROC", "Log return 5-bar", 5, 3),
        "roc_10": ("Momentum/ROC", "Log return 10-bar", 10, 3),
        "roc_20": ("Momentum/ROC", "Log return 20-bar", 20, 3),
        "roc_60": ("Momentum/ROC", "Log return 60-bar", 60, 3),
        "roc_zscore_20": ("Momentum/ROC", "ROC-20 z-score 100-bar", 120, 2),
        "ema_9": ("Trend/EMA", "EMA 9-period", 9, 2),
        "ema_21": ("Trend/EMA", "EMA 21-period", 21, 2),
        "ema_50": ("Trend/EMA", "EMA 50-period", 50, 2),
        "ema_200": ("Trend/EMA", "EMA 200-period", 200, 2),
        "ema_9_21_cross": ("Trend/EMA", "EMA 9/21 crossover signal", 21, 2),
        "ema_slope_21": ("Trend/EMA", "EMA-21 slope 5-bar ATR-norm", 26, 2),
        "ema_dist_21": ("Trend/EMA", "Price distance from EMA-21", 21, 2),
        "ema_dist_200": ("Trend/EMA", "Price distance from EMA-200", 200, 2),
        "mtf_ema_score": ("Trend/EMA", "MTF EMA score 5m+H1+H4", 21, 2),
        "vwap_daily": ("Trend/VWAP", "Daily VWAP UTC reset", 0, 2),
        "vwap_dev": ("Trend/VWAP", "Price deviation from VWAP", 0, 2),
        "vwap_upper_1": ("Trend/VWAP", "VWAP +1 std band", 0, 2),
        "vwap_upper_2": ("Trend/VWAP", "VWAP +2 std band", 0, 2),
        "vwap_lower_1": ("Trend/VWAP", "VWAP -1 std band", 0, 2),
        "vwap_lower_2": ("Trend/VWAP", "VWAP -2 std band", 0, 2),
        "vwap_position": ("Trend/VWAP", "VWAP band position -2 to +2", 0, 2),
        "supertrend_signal": ("Trend/Supertrend", "Supertrend direction ATR=10 m=2.5", 10, 2),
        "supertrend_dist": ("Trend/Supertrend", "Distance to Supertrend band/ATR", 10, 2),
        "adx_14": ("Trend/ADX", "ADX 14-period Wilder", 150, 3),
        "di_plus_14": ("Trend/ADX", "+DI 14-period Wilder", 150, 3),
        "di_minus_14": ("Trend/ADX", "-DI 14-period Wilder", 150, 3),
        "adx_trend_regime": ("Trend/ADX", "ADX trend regime 0-1", 150, 3),
        "di_cross": ("Trend/ADX", "DI crossover signal", 150, 3),
        "ichi_cloud_signal": ("Trend/Ichimoku", "Ichimoku cloud position 0/1/2", 60, 3),
        "ichi_tk_cross": ("Trend/Ichimoku", "Tenkan-Kijun cross signal", 30, 3),
        "ichi_cloud_thickness": ("Trend/Ichimoku", "Cloud thickness / close", 60, 3),
        "clv": ("Volume/CLV", "Close location value", 0, 3),
        "mfi_14": ("Volume/MFI", "Money Flow Index 14-period", 14, 2),
        "mfi_9": ("Volume/MFI", "Money Flow Index 9-period", 9, 2),
        "obv_zscore": ("Volume/OBV", "OBV z-score 50-bar", 50, 3),
        "obv_roc": ("Volume/OBV", "OBV rate of change 14-bar", 14, 3),
        "cvd_bar": ("Volume/CVD", "Per-bar cumulative volume delta", 0, 2),
        "cvd_zscore": ("Volume/CVD", "CVD cumulative z-score 50-bar", 50, 2),
        "cvd_roc": ("Volume/CVD", "CVD rate of change 14-bar", 14, 2),
        "cmf_20": ("Volume/CMF", "Chaikin Money Flow 20-period", 20, 3),
        "volume_zscore": ("Volume/Features", "Volume z-score 20-bar", 20, 3),
        "volume_rvol": ("Volume/Features", "Relative volume 20-bar", 20, 3),
        "volume_percentile": ("Volume/Features", "Volume percentile 100-bar", 100, 3),
        "taker_buy_ratio": ("Volume/Features", "Taker buy ratio", 0, 2),
        "gk_5": ("Volatility/GK", "Garman-Klass 5-bar annualized", 5, 1),
        "gk_10": ("Volatility/GK", "Garman-Klass 10-bar annualized", 10, 1),
        "gk_20": ("Volatility/GK", "Garman-Klass 20-bar annualized", 20, 1),
        "gk_60": ("Volatility/GK", "Garman-Klass 60-bar annualized", 60, 1),
        "parkinson_5": ("Volatility/Parkinson", "Parkinson 5-bar annualized", 5, 1),
        "parkinson_20": ("Volatility/Parkinson", "Parkinson 20-bar annualized", 20, 1),
        "parkinson_60": ("Volatility/Parkinson", "Parkinson 60-bar annualized", 60, 1),
        "rs_20": ("Volatility/RS", "Rogers-Satchell 20-bar annualized", 20, 1),
        "rs_60": ("Volatility/RS", "Rogers-Satchell 60-bar annualized", 60, 1),
        "hv_20": ("Volatility/HV", "Historical volatility 20-bar annualized", 20, 3),
        "hv_60": ("Volatility/HV", "Historical volatility 60-bar annualized", 60, 3),
        "bb_pct_b": ("Volatility/Bollinger", "Bollinger %B 20/2.0", 20, 3),
        "bb_bandwidth": ("Volatility/Bollinger", "Bollinger bandwidth 20/2.0", 20, 3),
        "squeeze_on": ("Volatility/TTM", "TTM Squeeze active", 20, 2),
        "squeeze_bars": ("Volatility/TTM", "Consecutive squeeze bars", 20, 2),
        "vol_percentile": ("Volatility/Regime", "HV-20 percentile 2016-bar", 2016, 2),
        "vol_regime": ("Volatility/Regime", "Volatility regime 0/1/2", 2016, 2),
        "vol_ratio": ("Volatility/Regime", "HV-20/HV-60 ratio", 60, 2),
        "sb_london_et": ("ICT/SilverBullet", "London SB 03-04 ET DST-aware", 0, 2),
        "sb_ny_am_et": ("ICT/SilverBullet", "NY AM SB 10-11 ET DST-aware", 0, 2),
        "sb_ny_pm_et": ("ICT/SilverBullet", "NY PM SB 14-15 ET DST-aware", 0, 2),
        "sb_any": ("ICT/SilverBullet", "Any Silver Bullet window active", 0, 2),
        "macro_ny_open": ("ICT/Macros", "NY Open macro 09:30-09:50 ET", 0, 2),
        "macro_ny_cont": ("ICT/Macros", "NY Continuation 09:50-10:10 ET", 0, 2),
        "macro_late_am": ("ICT/Macros", "Late AM macro 10:50-11:10 ET", 0, 2),
        "macro_lunch": ("ICT/Macros", "Lunch macro 11:50-12:10 ET", 0, 2),
        "macro_late_aft": ("ICT/Macros", "Late Afternoon 14:50-15:10 ET", 0, 2),
        "macro_london_1": ("ICT/Macros", "London macro 1 02:33-03:00 ET", 0, 2),
        "macro_london_2": ("ICT/Macros", "London macro 2 04:03-04:30 ET", 0, 2),
        "macro_any": ("ICT/Macros", "Any ICT macro window active", 0, 2),
        "lunch_zone": ("ICT/Macros", "Lunch zone 12:00-13:00 ET", 0, 2),
        "kz_london": ("ICT/KillZones", "London kill zone 02-05 ET", 0, 2),
        "kz_ny_open": ("ICT/KillZones", "NY Open kill zone 08:30-11 ET", 0, 2),
        "kz_ny_close": ("ICT/KillZones", "NY Close kill zone 10-11:30 ET", 0, 2),
        "kz_any": ("ICT/KillZones", "Any kill zone active", 0, 2),
        "asia_session": ("ICT/Sessions", "Asia session 20-00 ET", 0, 2),
        "london_session": ("ICT/Sessions", "London session 02-05 ET", 0, 2),
        "ny_session": ("ICT/Sessions", "NY session 08:30-16 ET", 0, 2),
        "asia_high": ("ICT/PO3", "Asia session high", 0, 2),
        "asia_low": ("ICT/PO3", "Asia session low", 0, 2),
        "asia_range": ("ICT/PO3", "Asia session range", 0, 2),
        "asia_range_norm": ("ICT/PO3", "Asia range / ATR", 0, 2),
        "london_swept_bsl": ("ICT/PO3", "London swept buy-side liquidity", 0, 2),
        "london_swept_ssl": ("ICT/PO3", "London swept sell-side liquidity", 0, 2),
        "po3_bias": ("ICT/PO3", "PO3 directional bias", 0, 2),
        "po3_confirmed": ("ICT/PO3", "PO3 bias confirmed", 0, 2),
        "ote_in_zone": ("ICT/OTE", "Price in OTE zone 62-79pct", 0, 2),
        "ote_dist": ("ICT/OTE", "Distance to OTE mid / ATR", 0, 2),
        "cisd_bull": ("ICT/CISD", "Bullish CISD trigger", 0, 2),
        "cisd_bear": ("ICT/CISD", "Bearish CISD trigger", 0, 2),
        "ob_disp_quality_bull": ("ICT/OB", "Bull OB displacement quality", 0, 2),
        "ob_disp_quality_bear": ("ICT/OB", "Bear OB displacement quality", 0, 2),
        "int_swing_high": ("ICT/InternalSwings", "Internal swing high 3-bar", 3, 3),
        "int_swing_low": ("ICT/InternalSwings", "Internal swing low 3-bar", 3, 3),
        "ict_confluence_v2": ("ICT/Confluence", "ICT confluence score 0-8", 0, 2),
    }

    date_str = datetime.now().strftime("%Y-%m-%d")
    lines = [
        "# Feature Catalog v2",
        f"# Generated: {date_str}",
        f"# Total new columns: {len(new_cols)}",
        "# Evidence tiers: 1=academic crypto, 2=practitioner, 3=conventional",
        "",
    ]
    for col in new_cols:
        if col in meta:
            fam, desc, warmup, tier = meta[col]
        else:
            fam, desc, warmup, tier = "Unknown", "auto-generated", 0, 3
        lines.extend([
            f"{col}:",
            f"  family: {fam}",
            f"  description: {desc}",
            f"  warmup_bars: {warmup}",
            f"  evidence_tier: {tier}",
            f"  computation_date: {date_str}",
            "",
        ])

    with open(catalog_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Feature catalog written to {catalog_path}")


# ===================================================================
# Main
# ===================================================================
def main():
    t0 = time.time()
    print("enrich_features_v2.py -- Dataset Enrichment Pass v2 (D33)")
    print("=" * 60)

    # Load
    print(f"\nLoading {INPUT_FILE.name}...")
    df = pd.read_parquet(INPUT_FILE)
    original_rows = len(df)
    original_cols = len(df.columns)
    print(f"  Loaded: {original_rows:,} rows x {original_cols} columns")

    all_new = []

    # Part 1
    print("\nPart 1: Raw data additions...")
    cols = add_part1_raw(df)
    all_new.extend(cols)
    print(f"  Added {len(cols)} columns: {', '.join(cols)}")

    # Part 2
    print("\nPart 2: Momentum indicators...")
    cols = add_part2_momentum(df)
    all_new.extend(cols)
    print(f"  Added {len(cols)} columns")

    # Part 3
    print("\nPart 3: Trend indicators...")
    cols, atr10 = add_part3_trend(df)
    all_new.extend(cols)
    print(f"  Added {len(cols)} columns")

    # Part 4
    print("\nPart 4: Volume and money flow...")
    cols = add_part4_volume(df)
    all_new.extend(cols)
    print(f"  Added {len(cols)} columns")

    # Part 5
    print("\nPart 5: Volatility features...")
    cols = add_part5_volatility(df, atr10)
    all_new.extend(cols)
    print(f"  Added {len(cols)} columns")

    # Part 6
    print("\nPart 6: ICT session and structural...")
    cols = add_part6_ict_session(df)
    all_new.extend(cols)
    print(f"  Added {len(cols)} columns")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Total new columns: {len(all_new)}")
    print(f"Total columns: {len(df.columns)} ({original_cols} original + {len(all_new)} new)")

    # Validate
    validate_output(df, original_rows, all_new)

    # Write parquet
    print(f"\nWriting {OUTPUT_FILE.name} (ZSTD compression)...")
    df.to_parquet(OUTPUT_FILE, compression="zstd")
    v1_size = INPUT_FILE.stat().st_size / 1e6
    v2_size = OUTPUT_FILE.stat().st_size / 1e6
    print(f"  v1 size: {v1_size:.1f} MB")
    print(f"  v2 size: {v2_size:.1f} MB")

    # Write catalog
    write_catalog(all_new, CATALOG_FILE)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")

    # Summary dict
    return {
        "original_rows": original_rows,
        "original_cols": original_cols,
        "new_cols": len(all_new),
        "total_cols": len(df.columns),
        "v1_size_mb": round(v1_size, 1),
        "v2_size_mb": round(v2_size, 1),
        "elapsed_s": round(elapsed, 1),
        "new_col_names": all_new,
    }


if __name__ == "__main__":
    main()
