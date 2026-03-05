# core/signals/ict/rules.py
# Standalone causal ICT signal functions.
# D39 -- ported from legacy/scripts/enrich_ict_v4.py and enrich_features_v2.py.
# D42 -- added compute_ob_quality (RQ1: OB quality score).
# D44 -- added detect_breaker_blocks (RQ4: breaker block encoding).
# D53 -- ICT rules overhaul: displacement, dual-layer swings, anchored OBs,
#         enhanced FVGs (CE/fill/IFVG), sweeps, sweep sequence, corrected CISD,
#         continuous premium/discount, OTE 0.705, MSS. 18 causality tests PASS.
# ASCII-safe for cp1252.
#
# CAUSALITY: Every function uses only df.iloc[:T+1] at bar T.
# Swing points use delayed confirmation (pivot_n bars delay).
# FVG detection uses the completed 3-bar pattern (bars i-2, i-1, i).
# OB detection triggers at BOS event, looks backward only.
#
# DEPENDENCY ORDER (D53):
#   1. detect_displacement (standalone, OHLC + ATR)
#   2. compute_swing_points / compute_swing_dual_layer (standalone)
#   3. detect_ob_bull/bear, detect_ob_anchored (need bos_close)
#   4. detect_fvg_bull/bear, detect_fvg_enhanced (need displacement)
#   5. compute_liq_levels (needs swing prices)
#   6. detect_sweep (needs int_swing from dual layer)
#   7. detect_sweep_sequence (needs sweep + displacement + FVG)
#   8. compute_premium_discount (needs ext_swing prices)
#   9. compute_ote_dist (needs swing prices)
#  10. compute_cisd (needs sweep ages)
#  11. detect_mss (needs int_choch + displacement + sweep)
#  12. detect_breaker_blocks (needs bos_close)
#  13. compute_ob_quality (needs bos_close + volume)

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------
def _ensure_atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    """Return ATR array; compute from OHLC if ict_atr_14 is missing."""
    if "ict_atr_14" in df.columns:
        return df["ict_atr_14"].values.copy()
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    pc = np.roll(c, 1)
    pc[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    atr = pd.Series(tr).ewm(alpha=1 / period, min_periods=period).mean().values
    return atr


# ---------------------------------------------------------------------------
# Displacement constants (D53 Task F -- searchable)
# ---------------------------------------------------------------------------
DISP_K = 1.5                # body >= 1.5x ATR
DISP_CLOSE_FRAC = 0.75      # close in top/bottom 25% of range


def _is_displacement_candle(o, h, l, c, atr, direction):
    """
    Check if a single candle qualifies as a displacement candle.
    Returns (occurred: bool, strength: float, close_loc: float).
    strength = body / ATR. close_loc = (c - l) / (h - l), 0..1.
    """
    rng = max(h - l, 1e-9)
    body = abs(c - o)
    strength = body / max(atr, 1e-9)
    close_loc = (c - l) / rng
    if direction == "bull":
        occurred = (c > o) and (strength >= DISP_K) and (close_loc >= DISP_CLOSE_FRAC)
    else:
        occurred = (c < o) and (strength >= DISP_K) and (close_loc <= (1.0 - DISP_CLOSE_FRAC))
    return occurred, strength, close_loc


def _ob_zone(o, h, l, c, direction, zone_mode="hybrid"):
    """Compute OB zone boundaries (D53 Task A)."""
    if zone_mode == "wick":
        return l, h
    if zone_mode == "body":
        return min(o, c), max(o, c)
    # hybrid
    if direction == "bull":  # OB candle is bearish
        return l, max(o, c)
    else:  # OB candle is bullish
        return min(o, c), h


def _ob_quality_score(disp_strength, bars_to_bos, width_atr,
                      has_fvg, state, age):
    """Quality score for OB top-3 ranking (D53 Task A)."""
    def _c01(x):
        return max(0.0, min(1.0, x))
    disp_term = _c01((disp_strength - 1.0) / 2.0)
    bos_term = _c01(1.0 - (bars_to_bos / 12.0))
    width_term = _c01(width_atr / 1.0)
    fvg_boost = 1.0 + 0.25 * float(has_fvg)
    fresh_mult = 1.0 if state == 1 else 0.85
    age_mult = np.exp(-age / 200.0)
    score = (0.45 * disp_term + 0.35 * bos_term + 0.20 * width_term)
    return score * fvg_boost * fresh_mult * age_mult


def _try_create_ob_anchored(opens, highs, lows, closes, atr, t,
                            direction, disp_search, anchor_search,
                            zone_mode):
    """Find displacement + OB candle backward from BOS at bar t."""
    # Step 1: find most recent displacement candle
    disp_idx = None
    disp_strength = 0.0
    for j in range(t, max(-1, t - disp_search), -1):
        a_j = atr[j]
        if np.isnan(a_j) or a_j <= 0:
            continue
        ok, strength, _ = _is_displacement_candle(
            opens[j], highs[j], lows[j], closes[j], a_j, direction)
        if ok:
            disp_idx = j
            disp_strength = strength
            break
    if disp_idx is None:
        return None

    # Step 2: find opposite-colored candle before displacement
    ob_idx = None
    for j in range(disp_idx - 1, max(-1, disp_idx - anchor_search), -1):
        if direction == "bull" and closes[j] < opens[j]:
            ob_idx = j
            break
        if direction == "bear" and closes[j] > opens[j]:
            ob_idx = j
            break
    if ob_idx is None:
        return None

    # Zone computation
    o, h, l, c = opens[ob_idx], highs[ob_idx], lows[ob_idx], closes[ob_idx]
    bot, top = _ob_zone(o, h, l, c, direction, zone_mode)

    # FVG check at displacement candle
    has_fvg = 0
    if disp_idx >= 2:
        if direction == "bull" and lows[disp_idx] > highs[disp_idx - 2]:
            has_fvg = 1
        elif direction == "bear" and highs[disp_idx] < lows[disp_idx - 2]:
            has_fvg = 1

    return {
        "ob_idx": ob_idx,
        "disp_idx": disp_idx,
        "disp_strength": disp_strength,
        "bot": bot,
        "top": top,
        "mid": (bot + top) / 2.0,
        "width": top - bot,
        "state": 1,
        "age": 0,
        "direction": direction,
        "bos_bar": t,
        "bars_to_bos": t - disp_idx,
        "has_fvg": has_fvg,
    }


# ---------------------------------------------------------------------------
# 0. detect_displacement (D53 Task F -- prerequisite for A, C, H, MSS)
# ---------------------------------------------------------------------------
def detect_displacement(
    df: pd.DataFrame,
    disp_k: float = 1.5,
    disp_close_frac: float = 0.75,
    age_cap: int = 48,
    multi_k_total: float = 2.0,
) -> pd.DataFrame:
    """
    Detect single and multi-candle displacement events at every bar.

    Single-candle: body >= disp_k * ATR, close in top/bottom
                   disp_close_frac of range.
    Multi-candle: 3 consecutive same-direction candles,
                  combined move >= multi_k_total * ATR.

    age_cap: bars until displacement properties go NaN.

    Returns DataFrame with 14 columns (7 per direction bull/bear):
      displacement_{dir}, displacement_{dir}_age,
      displacement_{dir}_strength, displacement_{dir}_close_loc,
      displacement_{dir}_range_atr, displacement_{dir}_has_fvg,
      displacement_{dir}_is_multi
    """
    size = len(df)
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    atr = _ensure_atr(df)

    # Output arrays -- 7 per direction
    ob_fire = np.zeros(size, dtype=np.float32)
    ob_age = np.full(size, np.nan, dtype=np.float32)
    ob_str = np.full(size, np.nan, dtype=np.float32)
    ob_cloc = np.full(size, np.nan, dtype=np.float32)
    ob_ratr = np.full(size, np.nan, dtype=np.float32)
    ob_fvg = np.full(size, np.nan, dtype=np.float32)
    ob_multi = np.full(size, np.nan, dtype=np.float32)

    or_fire = np.zeros(size, dtype=np.float32)
    or_age = np.full(size, np.nan, dtype=np.float32)
    or_str = np.full(size, np.nan, dtype=np.float32)
    or_cloc = np.full(size, np.nan, dtype=np.float32)
    or_ratr = np.full(size, np.nan, dtype=np.float32)
    or_fvg = np.full(size, np.nan, dtype=np.float32)
    or_multi = np.full(size, np.nan, dtype=np.float32)

    # Persistent state for bull displacement
    b_age = -1
    b_str = b_cloc = b_ratr = b_fvg = b_mlt = 0.0
    # Persistent state for bear displacement
    r_age = -1
    r_str = r_cloc = r_ratr = r_fvg = r_mlt = 0.0

    for i in range(size):
        o_i, h_i, l_i, c_i = opens[i], highs[i], lows[i], closes[i]
        a_i = atr[i]
        valid_atr = not (np.isnan(a_i) or a_i <= 0)
        if not valid_atr:
            a_i = 1e-9  # safe denominator for division

        rng = max(h_i - l_i, 1e-9)
        body = abs(c_i - o_i)
        strength = body / a_i
        close_loc = (c_i - l_i) / rng
        range_atr = rng / a_i

        # ---- BULL displacement ----
        single_b = (valid_atr and c_i > o_i and strength >= disp_k
                    and close_loc >= disp_close_frac)
        multi_b = False
        multi_b_str = 0.0
        if valid_atr and i >= 2:
            if (closes[i] > opens[i]
                    and closes[i - 1] > opens[i - 1]
                    and closes[i - 2] > opens[i - 2]):
                multi_b_str = abs(closes[i] - opens[i - 2]) / a_i
                if multi_b_str >= multi_k_total:
                    multi_b = True

        if single_b or multi_b:
            ob_fire[i] = 1.0
            b_age = 0
            b_str = strength if single_b else multi_b_str
            b_cloc = close_loc
            b_ratr = range_atr
            b_mlt = 0.0 if single_b else 1.0
            # FVG check: bull FVG at bar i = lows[i] > highs[i-2]
            b_fvg = 1.0 if (i >= 2 and lows[i] > highs[i - 2]) else 0.0
        elif b_age >= 0:
            b_age += 1

        if 0 <= b_age <= age_cap:
            ob_age[i] = float(b_age)
            ob_str[i] = b_str
            ob_cloc[i] = b_cloc
            ob_ratr[i] = b_ratr
            ob_fvg[i] = b_fvg
            ob_multi[i] = b_mlt
        elif b_age > age_cap:
            b_age = -1

        # ---- BEAR displacement ----
        single_r = (valid_atr and c_i < o_i and strength >= disp_k
                    and close_loc <= (1.0 - disp_close_frac))
        multi_r = False
        multi_r_str = 0.0
        if valid_atr and i >= 2:
            if (closes[i] < opens[i]
                    and closes[i - 1] < opens[i - 1]
                    and closes[i - 2] < opens[i - 2]):
                multi_r_str = abs(closes[i] - opens[i - 2]) / a_i
                if multi_r_str >= multi_k_total:
                    multi_r = True

        if single_r or multi_r:
            or_fire[i] = 1.0
            r_age = 0
            r_str = strength if single_r else multi_r_str
            r_cloc = close_loc
            r_ratr = range_atr
            r_mlt = 0.0 if single_r else 1.0
            # FVG check: bear FVG at bar i = highs[i] < lows[i-2]
            r_fvg = 1.0 if (i >= 2 and highs[i] < lows[i - 2]) else 0.0
        elif r_age >= 0:
            r_age += 1

        if 0 <= r_age <= age_cap:
            or_age[i] = float(r_age)
            or_str[i] = r_str
            or_cloc[i] = r_cloc
            or_ratr[i] = r_ratr
            or_fvg[i] = r_fvg
            or_multi[i] = r_mlt
        elif r_age > age_cap:
            r_age = -1

    return pd.DataFrame({
        "displacement_bull": ob_fire,
        "displacement_bull_age": ob_age,
        "displacement_bull_strength": ob_str,
        "displacement_bull_close_loc": ob_cloc,
        "displacement_bull_range_atr": ob_ratr,
        "displacement_bull_has_fvg": ob_fvg,
        "displacement_bull_is_multi": ob_multi,
        "displacement_bear": or_fire,
        "displacement_bear_age": or_age,
        "displacement_bear_strength": or_str,
        "displacement_bear_close_loc": or_cloc,
        "displacement_bear_range_atr": or_ratr,
        "displacement_bear_has_fvg": or_fvg,
        "displacement_bear_is_multi": or_multi,
    }, index=df.index)


# ---------------------------------------------------------------------------
# 1. compute_swing_points
# ---------------------------------------------------------------------------
def compute_swing_points(
    df: pd.DataFrame,
    pivot_n: int = 3,
) -> pd.DataFrame:
    """
    Causal swing high/low detection with delayed confirmation.

    At bar i, swing at bar i-pivot_n is confirmed if it is the extreme
    of the window [i-2*pivot_n, i] (all data up to current bar).

    BOS/CHoCH and market_trend computed inline from confirmed swings.

    Returns DataFrame (same index as df) with columns:
      swing_high, swing_low, swing_high_price, swing_low_price,
      market_trend, bos_close, bos_wick, choch_close
    """
    size = len(df)
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    n = pivot_n

    swing_high = np.zeros(size, dtype=np.int8)
    swing_low = np.zeros(size, dtype=np.int8)
    swing_high_price = np.full(size, np.nan)
    swing_low_price = np.full(size, np.nan)
    bos_close_arr = np.zeros(size, dtype=np.int8)
    bos_wick_arr = np.zeros(size, dtype=np.int8)
    choch_close_arr = np.zeros(size, dtype=np.int8)
    trend_arr = np.zeros(size, dtype=np.int8)

    last_sh = np.nan
    last_sl = np.nan
    current_trend = 0

    for i in range(size):
        # 1. Confirm swing at bar i - n (causal delayed confirmation)
        cand = i - n
        if cand >= n:
            w_start = cand - n
            w_end = i + 1  # exclusive, so window is [cand-n, i]
            wh = highs[w_start:w_end]
            wl = lows[w_start:w_end]
            if highs[cand] >= np.max(wh):
                swing_high[cand] = 1
                swing_high_price[cand] = highs[cand]
                last_sh = highs[cand]
            if lows[cand] <= np.min(wl):
                swing_low[cand] = 1
                swing_low_price[cand] = lows[cand]
                last_sl = lows[cand]

        # 2. Close-based BOS / CHoCH
        if not np.isnan(last_sh) and closes[i] > last_sh:
            if current_trend == 1:
                bos_close_arr[i] = 1
            elif current_trend == -1:
                choch_close_arr[i] = 1
            current_trend = 1
        elif not np.isnan(last_sl) and closes[i] < last_sl:
            if current_trend == -1:
                bos_close_arr[i] = -1
            elif current_trend == 1:
                choch_close_arr[i] = -1
            current_trend = -1

        # 3. Wick-based BOS (directional flag only)
        if not np.isnan(last_sh) and highs[i] > last_sh:
            bos_wick_arr[i] = 1
        if not np.isnan(last_sl) and lows[i] < last_sl:
            bos_wick_arr[i] = -1

        trend_arr[i] = current_trend

    return pd.DataFrame({
        "swing_high": swing_high,
        "swing_low": swing_low,
        "swing_high_price": swing_high_price,
        "swing_low_price": swing_low_price,
        "market_trend": trend_arr,
        "bos_close": bos_close_arr,
        "bos_wick": bos_wick_arr,
        "choch_close": choch_close_arr,
    }, index=df.index)


# ---------------------------------------------------------------------------
# 1b. compute_swing_dual_layer (D53 Task D)
# ---------------------------------------------------------------------------
def compute_swing_dual_layer(
    df: pd.DataFrame,
    pivot_n_internal: int = 5,
    pivot_n_external: int = 10,
) -> pd.DataFrame:
    """
    Dual-layer swing detection: internal (N=5, entry timing, 25 min lag)
    and external (N=10, structure/dealing range, 50 min lag).

    Each layer independently confirms swings, computes BOS/CHoCH,
    and forward-fills swing prices.

    Returns DataFrame with 22 columns (11 per layer, int_ and ext_ prefix):
      {prefix}_swing_high, {prefix}_swing_low,
      {prefix}_swing_high_price, {prefix}_swing_low_price,
      {prefix}_dist_to_sh_atr, {prefix}_dist_to_sl_atr,
      {prefix}_trend, {prefix}_bos_bull, {prefix}_bos_bear,
      {prefix}_choch_bull, {prefix}_choch_bear
    """
    size = len(df)
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    atr = _ensure_atr(df)

    def _compute_layer(n):
        """Compute swing/BOS/CHoCH for one layer with pivot_n = n."""
        sh_event = np.zeros(size, dtype=np.float32)
        sl_event = np.zeros(size, dtype=np.float32)
        sh_price = np.full(size, np.nan, dtype=np.float32)
        sl_price = np.full(size, np.nan, dtype=np.float32)
        dist_sh = np.full(size, np.nan, dtype=np.float32)
        dist_sl = np.full(size, np.nan, dtype=np.float32)
        trend = np.zeros(size, dtype=np.float32)
        bos_bull = np.zeros(size, dtype=np.float32)
        bos_bear = np.zeros(size, dtype=np.float32)
        choch_bull = np.zeros(size, dtype=np.float32)
        choch_bear = np.zeros(size, dtype=np.float32)

        last_sh = np.nan
        last_sl = np.nan
        cur_trend = 0

        for i in range(size):
            # Confirm swing at candidate bar i - n
            cand = i - n
            if cand >= n:
                w_start = cand - n
                w_end = i + 1  # window [cand-n, i]
                wh = highs[w_start:w_end]
                wl = lows[w_start:w_end]
                if highs[cand] >= np.max(wh):
                    sh_event[i] = 1.0
                    last_sh = highs[cand]
                if lows[cand] <= np.min(wl):
                    sl_event[i] = 1.0
                    last_sl = lows[cand]

            # Forward-fill prices
            sh_price[i] = last_sh
            sl_price[i] = last_sl

            # Distance to swing levels (ATR-normalized)
            a = atr[i]
            if not np.isnan(last_sh) and not np.isnan(a) and a > 0:
                dist_sh[i] = (last_sh - closes[i]) / a
            if not np.isnan(last_sl) and not np.isnan(a) and a > 0:
                dist_sl[i] = (closes[i] - last_sl) / a

            # BOS / CHoCH (close-based, independent per layer)
            if not np.isnan(last_sh) and closes[i] > last_sh:
                if cur_trend == 1:
                    bos_bull[i] = 1.0
                elif cur_trend == -1:
                    choch_bull[i] = 1.0
                cur_trend = 1
            elif not np.isnan(last_sl) and closes[i] < last_sl:
                if cur_trend == -1:
                    bos_bear[i] = 1.0
                elif cur_trend == 1:
                    choch_bear[i] = 1.0
                cur_trend = -1

            trend[i] = float(cur_trend)

        return (sh_event, sl_event, sh_price, sl_price,
                dist_sh, dist_sl, trend, bos_bull, bos_bear,
                choch_bull, choch_bear)

    # Compute both layers
    int_result = _compute_layer(pivot_n_internal)
    ext_result = _compute_layer(pivot_n_external)

    names = ["swing_high", "swing_low", "swing_high_price", "swing_low_price",
             "dist_to_sh_atr", "dist_to_sl_atr", "trend",
             "bos_bull", "bos_bear", "choch_bull", "choch_bear"]

    cols = {}
    for idx, name in enumerate(names):
        cols[f"int_{name}"] = int_result[idx]
        cols[f"ext_{name}"] = ext_result[idx]

    return pd.DataFrame(cols, index=df.index)


# ---------------------------------------------------------------------------
# 2. detect_ob_bull
# ---------------------------------------------------------------------------
def detect_ob_bull(
    df: pd.DataFrame,
    lookback: int = 200,
) -> pd.DataFrame:
    """
    Bull order block: last bearish candle before a bullish BOS (close version).
    Detected at the BOS event bar, not retroactively.

    Requires: bos_close column in df (from compute_swing_points).

    Returns DataFrame with columns:
      ob_bull_age, ob_bull_top, ob_bull_bot, ob_bull_mid, ob_bull_in_zone
    """
    size = len(df)
    opens = df["open"].values
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    bos = df["bos_close"].values

    out_top = np.full(size, np.nan)
    out_bot = np.full(size, np.nan)
    out_mid = np.full(size, np.nan)
    out_age = np.full(size, np.nan)
    out_zone = np.zeros(size, dtype=np.int8)

    curr_top = curr_bot = np.nan
    ob_active = False
    age = 0

    for i in range(size):
        # Detect new bull OB at bullish BOS
        if bos[i] == 1:
            found = False
            search_start = max(0, i - lookback)
            for j in range(i - 1, search_start - 1, -1):
                if closes[j] < opens[j]:
                    curr_top = opens[j]   # body top of red candle
                    curr_bot = closes[j]  # body bot of red candle
                    ob_active = True
                    age = 0
                    found = True
                    break
            if not found:
                ob_active = False

        if ob_active:
            out_top[i] = curr_top
            out_bot[i] = curr_bot
            out_mid[i] = (curr_top + curr_bot) / 2
            out_age[i] = age
            out_zone[i] = 1 if (lows[i] <= curr_top and highs[i] >= curr_bot) else 0
            age += 1

    return pd.DataFrame({
        "ob_bull_age": out_age,
        "ob_bull_top": out_top,
        "ob_bull_bot": out_bot,
        "ob_bull_mid": out_mid,
        "ob_bull_in_zone": out_zone,
    }, index=df.index)


# ---------------------------------------------------------------------------
# 3. detect_ob_bear
# ---------------------------------------------------------------------------
def detect_ob_bear(
    df: pd.DataFrame,
    lookback: int = 200,
) -> pd.DataFrame:
    """
    Bear order block: last bullish candle before a bearish BOS (close version).

    Requires: bos_close column in df (from compute_swing_points).

    Returns DataFrame with columns:
      ob_bear_age, ob_bear_top, ob_bear_bot, ob_bear_mid, ob_bear_in_zone
    """
    size = len(df)
    opens = df["open"].values
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    bos = df["bos_close"].values

    out_top = np.full(size, np.nan)
    out_bot = np.full(size, np.nan)
    out_mid = np.full(size, np.nan)
    out_age = np.full(size, np.nan)
    out_zone = np.zeros(size, dtype=np.int8)

    curr_top = curr_bot = np.nan
    ob_active = False
    age = 0

    for i in range(size):
        if bos[i] == -1:
            found = False
            search_start = max(0, i - lookback)
            for j in range(i - 1, search_start - 1, -1):
                if closes[j] > opens[j]:  # bullish candle
                    curr_top = closes[j]  # body top of green candle
                    curr_bot = opens[j]   # body bot of green candle
                    ob_active = True
                    age = 0
                    found = True
                    break
            if not found:
                ob_active = False

        if ob_active:
            out_top[i] = curr_top
            out_bot[i] = curr_bot
            out_mid[i] = (curr_top + curr_bot) / 2
            out_age[i] = age
            out_zone[i] = 1 if (lows[i] <= curr_top and highs[i] >= curr_bot) else 0
            age += 1

    return pd.DataFrame({
        "ob_bear_age": out_age,
        "ob_bear_top": out_top,
        "ob_bear_bot": out_bot,
        "ob_bear_mid": out_mid,
        "ob_bear_in_zone": out_zone,
    }, index=df.index)


# ---------------------------------------------------------------------------
# 3b. detect_ob_anchored (D53 Task A -- displacement-anchored, 3-state)
# ---------------------------------------------------------------------------
def detect_ob_anchored(
    df: pd.DataFrame,
    ob_disp_search: int = 30,
    ob_anchor_search: int = 20,
    ob_age_cap: int = 200,
    mit_p: float = 0.5,
    zone_mode: str = "hybrid",
) -> pd.DataFrame:
    """
    Displacement-anchored OB detection with 3-state tracking.

    Algorithm:
      1. At BOS event, search backward for displacement candle
      2. From displacement, search backward for opposite-colored OB candle
      3. Track OBs: fresh(1) -> mitigated(2) -> invalid(remove)
      4. Output top-3 OBs per direction ranked by quality score

    Requires: bos_close column in df (from compute_swing_points).

    Returns DataFrame with 82 columns:
      ob_{bull,bear}_{1,2,3}_{state,age,top,bot,mid,width_atr,in_zone,
        penetration,dist_top_atr,dist_bot_atr,strength,bars_to_bos,has_fvg}
      count_active_ob_{bull,bear}, min_dist_ob_{bull,bear}_atr
    """
    size = len(df)
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    bos = df["bos_close"].values
    atr = _ensure_atr(df)

    # Pre-allocate output arrays (82 columns)
    out = {}
    for d in ("bull", "bear"):
        for k in range(1, 4):
            p = f"ob_{d}_{k}"
            out[f"{p}_state"] = np.zeros(size, dtype=np.float32)
            out[f"{p}_age"] = np.full(size, np.nan, dtype=np.float32)
            out[f"{p}_top"] = np.full(size, np.nan, dtype=np.float32)
            out[f"{p}_bot"] = np.full(size, np.nan, dtype=np.float32)
            out[f"{p}_mid"] = np.full(size, np.nan, dtype=np.float32)
            out[f"{p}_width_atr"] = np.full(size, np.nan, dtype=np.float32)
            out[f"{p}_in_zone"] = np.zeros(size, dtype=np.float32)
            out[f"{p}_penetration"] = np.full(size, np.nan, dtype=np.float32)
            out[f"{p}_dist_top_atr"] = np.full(size, np.nan, dtype=np.float32)
            out[f"{p}_dist_bot_atr"] = np.full(size, np.nan, dtype=np.float32)
            out[f"{p}_strength"] = np.full(size, np.nan, dtype=np.float32)
            out[f"{p}_bars_to_bos"] = np.full(size, np.nan, dtype=np.float32)
            out[f"{p}_has_fvg"] = np.full(size, np.nan, dtype=np.float32)
        out[f"count_active_ob_{d}"] = np.zeros(size, dtype=np.float32)
        out[f"min_dist_ob_{d}_atr"] = np.full(size, np.nan, dtype=np.float32)

    bull_obs = []
    bear_obs = []

    for i in range(size):
        h_i, l_i, c_i = highs[i], lows[i], closes[i]
        a_i = atr[i]
        va = not (np.isnan(a_i) or a_i <= 0)
        if not va:
            a_i = 1e-9

        # --- Create new OBs at BOS ---
        if bos[i] == 1:
            ob = _try_create_ob_anchored(
                opens, highs, lows, closes, atr, i, "bull",
                ob_disp_search, ob_anchor_search, zone_mode)
            if ob is not None:
                bull_obs.append(ob)
        elif bos[i] == -1:
            ob = _try_create_ob_anchored(
                opens, highs, lows, closes, atr, i, "bear",
                ob_disp_search, ob_anchor_search, zone_mode)
            if ob is not None:
                bear_obs.append(ob)

        # --- Update and filter bull OBs ---
        new_bull = []
        for ob in bull_obs:
            top, bot = ob["top"], ob["bot"]
            w = max(top - bot, 1e-9)
            if ob["state"] == 1:
                pen = max(0.0, min(1.0, (top - min(l_i, top)) / w))
                if pen >= mit_p:
                    ob["state"] = 2
            if ob["state"] == 2 and c_i < bot:
                continue
            ob["age"] += 1
            if ob["age"] > ob_age_cap:
                continue
            new_bull.append(ob)
        bull_obs = new_bull

        # --- Update and filter bear OBs ---
        new_bear = []
        for ob in bear_obs:
            top, bot = ob["top"], ob["bot"]
            w = max(top - bot, 1e-9)
            if ob["state"] == 1:
                pen = max(0.0, min(1.0, (max(h_i, bot) - bot) / w))
                if pen >= mit_p:
                    ob["state"] = 2
            if ob["state"] == 2 and c_i > top:
                continue
            ob["age"] += 1
            if ob["age"] > ob_age_cap:
                continue
            new_bear.append(ob)
        bear_obs = new_bear

        # --- Rank and output top-3 per direction ---
        for direction, obs_list in [("bull", bull_obs), ("bear", bear_obs)]:
            out[f"count_active_ob_{direction}"][i] = float(len(obs_list))
            if not obs_list:
                continue

            ranked = []
            for ob in obs_list:
                w_atr = (ob["top"] - ob["bot"]) / a_i if va else 0.0
                q = _ob_quality_score(
                    ob["disp_strength"], ob["bars_to_bos"],
                    w_atr, ob["has_fvg"], ob["state"], ob["age"])
                d_atr = abs(c_i - ob["mid"]) / a_i if va else 0.0
                ranked.append((q, d_atr, ob))
            ranked.sort(key=lambda x: (-x[0], x[1]))

            if va:
                out[f"min_dist_ob_{direction}_atr"][i] = min(
                    abs(c_i - ob["mid"]) / a_i for ob in obs_list)

            for k in range(min(3, len(ranked))):
                _, _, ob = ranked[k]
                p = f"ob_{direction}_{k + 1}"
                top, bot = ob["top"], ob["bot"]
                w = max(top - bot, 1e-9)

                out[f"{p}_state"][i] = float(ob["state"])
                out[f"{p}_age"][i] = min(ob["age"], ob_age_cap) / ob_age_cap
                out[f"{p}_top"][i] = top
                out[f"{p}_bot"][i] = bot
                out[f"{p}_mid"][i] = ob["mid"]
                out[f"{p}_width_atr"][i] = (
                    (top - bot) / a_i if va else np.nan)
                out[f"{p}_in_zone"][i] = (
                    1.0 if (l_i <= top and h_i >= bot) else 0.0)

                if direction == "bull":
                    pen_cur = max(0.0, min(1.0,
                        (top - min(l_i, top)) / w))
                else:
                    pen_cur = max(0.0, min(1.0,
                        (max(h_i, bot) - bot) / w))
                out[f"{p}_penetration"][i] = pen_cur

                out[f"{p}_dist_top_atr"][i] = (
                    (top - c_i) / a_i if va else np.nan)
                out[f"{p}_dist_bot_atr"][i] = (
                    (c_i - bot) / a_i if va else np.nan)
                out[f"{p}_strength"][i] = ob["disp_strength"]
                out[f"{p}_bars_to_bos"][i] = float(ob["bars_to_bos"])
                out[f"{p}_has_fvg"][i] = float(ob["has_fvg"])

    return pd.DataFrame(out, index=df.index)


# ---------------------------------------------------------------------------
# 4. detect_fvg_bull
# ---------------------------------------------------------------------------
def detect_fvg_bull(
    df: pd.DataFrame,
    age_cap: int = 100,
    min_size_atr: float = 0.50,
) -> pd.DataFrame:
    """
    Bull FVG detection using causal 3-bar pattern.

    Bull FVG at bar i: lows[i] > highs[i-2] (gap between bar i-2 high
    and bar i low). Confirmed at bar i (all three bars are known).

    Tracks active FVGs with age-based expiry (D13) and close-through
    mitigation (D14: mitigated when close <= fvg_top).

    Returns DataFrame with columns:
      fvg_bull_in_zone, fvg_bull_near_top, fvg_bull_near_bot,
      fvg_bull_age, fvg_bull_count
    """
    size = len(df)
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    atr = _ensure_atr(df)

    out_in_zone = np.zeros(size, dtype=np.int8)
    out_near_top = np.full(size, np.nan)
    out_near_bot = np.full(size, np.nan)
    out_age = np.full(size, np.nan)
    out_count = np.zeros(size, dtype=np.int16)

    active = []  # list of dicts {top, bot, age}

    for i in range(size):
        # Age-based expiry (D13)
        active = [f for f in active if f["age"] < age_cap]
        # Close-through mitigation (D14): bull FVG dies when close <= top
        active = [f for f in active if not (closes[i] <= f["top"])]

        # Detect new bull FVG at bar i (3-bar pattern: i-2, i-1, i)
        if i >= 2:
            gap = lows[i] - highs[i - 2]
            min_sz = atr[i] * min_size_atr if not np.isnan(atr[i]) else 0
            if gap > min_sz:
                active.append({
                    "top": lows[i],
                    "bot": highs[i - 2],
                    "age": 0,
                })

        for f in active:
            f["age"] += 1

        c = closes[i]
        h = highs[i]
        lo = lows[i]
        out_count[i] = len(active)

        if active:
            out_age[i] = active[-1]["age"]
            nearest = min(active, key=lambda f: min(abs(f["top"] - c), abs(f["bot"] - c)))
            out_near_top[i] = nearest["top"]
            out_near_bot[i] = nearest["bot"]
            if any(lo <= f["top"] and h >= f["bot"] for f in active):
                out_in_zone[i] = 1

    return pd.DataFrame({
        "fvg_bull_in_zone": out_in_zone,
        "fvg_bull_near_top": out_near_top,
        "fvg_bull_near_bot": out_near_bot,
        "fvg_bull_age": out_age,
        "fvg_bull_count": out_count,
    }, index=df.index)


# ---------------------------------------------------------------------------
# 5. detect_fvg_bear
# ---------------------------------------------------------------------------
def detect_fvg_bear(
    df: pd.DataFrame,
    age_cap: int = 100,
    min_size_atr: float = 0.50,
) -> pd.DataFrame:
    """
    Bear FVG detection using causal 3-bar pattern.

    Bear FVG at bar i: lows[i-2] > highs[i] (gap between bar i-2 low
    and bar i high). Confirmed at bar i.

    Mitigation: close >= fvg_bot.

    Returns DataFrame with columns:
      fvg_bear_in_zone, fvg_bear_near_top, fvg_bear_near_bot,
      fvg_bear_age, fvg_bear_count
    """
    size = len(df)
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    atr = _ensure_atr(df)

    out_in_zone = np.zeros(size, dtype=np.int8)
    out_near_top = np.full(size, np.nan)
    out_near_bot = np.full(size, np.nan)
    out_age = np.full(size, np.nan)
    out_count = np.zeros(size, dtype=np.int16)

    active = []

    for i in range(size):
        active = [f for f in active if f["age"] < age_cap]
        active = [f for f in active if not (closes[i] >= f["bot"])]

        if i >= 2:
            gap = lows[i - 2] - highs[i]
            min_sz = atr[i] * min_size_atr if not np.isnan(atr[i]) else 0
            if gap > min_sz:
                active.append({
                    "top": lows[i - 2],
                    "bot": highs[i],
                    "age": 0,
                })

        for f in active:
            f["age"] += 1

        c = closes[i]
        h = highs[i]
        lo = lows[i]
        out_count[i] = len(active)

        if active:
            out_age[i] = active[-1]["age"]
            nearest = min(active, key=lambda f: min(abs(f["top"] - c), abs(f["bot"] - c)))
            out_near_top[i] = nearest["top"]
            out_near_bot[i] = nearest["bot"]
            if any(lo <= f["top"] and h >= f["bot"] for f in active):
                out_in_zone[i] = 1

    return pd.DataFrame({
        "fvg_bear_in_zone": out_in_zone,
        "fvg_bear_near_top": out_near_top,
        "fvg_bear_near_bot": out_near_bot,
        "fvg_bear_age": out_age,
        "fvg_bear_count": out_count,
    }, index=df.index)


# ---------------------------------------------------------------------------
# 5b. detect_fvg_enhanced (D53 Task C -- top-3, CE, fill, IFVG)
# ---------------------------------------------------------------------------
IFVG_AGE_CAP = 144  # bars (~12h)


def _fvg_rank_score(dist_atr, age, width_atr, is_disp_fvg):
    """Ranking score for FVG top-3 selection."""
    import math
    dist_term = 1.0 / (1.0 + dist_atr)
    age_term = math.exp(-age / 288.0)
    size_term = max(0.0, min(1.0, width_atr / 1.0))
    disp_boost = 1.0 + 0.35 * float(is_disp_fvg)
    return disp_boost * (0.55 * dist_term + 0.30 * age_term + 0.15 * size_term)


def detect_fvg_enhanced(
    df: pd.DataFrame,
    age_cap: int = 100,
    min_size_atr: float = 0.50,
    ifvg_age_cap: int = 144,
) -> pd.DataFrame:
    """
    Enhanced FVG detection with top-3 tracking, CE levels, fill fractions,
    displacement tags, CE rejection, and IFVG (inverted FVG) tracking.

    Returns DataFrame with columns per direction (bull/bear) per rank (1/2/3):
      fvg_{dir}_{k}_ce, fvg_{dir}_{k}_dist_to_ce_atr,
      fvg_{dir}_{k}_ce_touched, fvg_{dir}_{k}_ce_rejected,
      fvg_{dir}_{k}_fill_fraction, fvg_{dir}_{k}_fully_filled,
      fvg_{dir}_{k}_is_displacement, fvg_{dir}_{k}_is_ifvg
    Plus aggregates: fvg_{dir}_count, fvg_{dir}_recent_age
    """
    size = len(df)
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    atr = _ensure_atr(df)

    # Check if displacement data is available
    has_disp = "displacement_bull" in df.columns
    if has_disp:
        disp_bull = df["displacement_bull"].values
        disp_bear = df["displacement_bear"].values

    # Pre-allocate output
    out = {}
    for d in ("bull", "bear"):
        for k in range(1, 4):
            p = f"fvg_{d}_{k}"
            out[f"{p}_ce"] = np.full(size, np.nan, dtype=np.float32)
            out[f"{p}_dist_to_ce_atr"] = np.full(size, np.nan, dtype=np.float32)
            out[f"{p}_ce_touched"] = np.zeros(size, dtype=np.float32)
            out[f"{p}_ce_rejected"] = np.zeros(size, dtype=np.float32)
            out[f"{p}_fill_fraction"] = np.zeros(size, dtype=np.float32)
            out[f"{p}_fully_filled"] = np.zeros(size, dtype=np.float32)
            out[f"{p}_is_displacement"] = np.zeros(size, dtype=np.float32)
            out[f"{p}_is_ifvg"] = np.zeros(size, dtype=np.float32)
        out[f"fvg_{d}_count"] = np.zeros(size, dtype=np.float32)
        out[f"fvg_{d}_recent_age"] = np.full(size, np.nan, dtype=np.float32)

    # Active FVG lists: each FVG is a dict
    bull_fvgs = []
    bear_fvgs = []

    for i in range(size):
        a_i = atr[i]
        va = not (np.isnan(a_i) or a_i <= 0)
        if not va:
            a_i = 1e-9
        c_i = closes[i]
        h_i = highs[i]
        l_i = lows[i]

        # --- Age-based expiry ---
        bull_fvgs = [f for f in bull_fvgs if f["age"] < age_cap]
        bear_fvgs = [f for f in bear_fvgs if f["age"] < age_cap]

        # --- Close-through mitigation ---
        # Bull FVG dies when close <= fvg_top (D14)
        new_bull = []
        for f in bull_fvgs:
            if c_i <= f["top"]:
                # Check if this becomes an IFVG (bear)
                if c_i < f["bot"] and f["age"] <= ifvg_age_cap and not f["is_ifvg"]:
                    f["is_ifvg"] = 1
                    f["direction"] = "bear_ifvg"
                    bear_fvgs.append(f)
                continue
            new_bull.append(f)
        bull_fvgs = new_bull

        # Bear FVG dies when close >= fvg_bot
        new_bear = []
        for f in bear_fvgs:
            if f.get("direction") == "bear_ifvg":
                # IFVG from bull: invalid if close goes back above top
                if c_i > f["top"]:
                    continue
                new_bear.append(f)
            else:
                if c_i >= f["bot"]:
                    # Check if becomes IFVG (bull)
                    if c_i > f["top"] and f["age"] <= ifvg_age_cap and not f["is_ifvg"]:
                        f["is_ifvg"] = 1
                        f["direction"] = "bull_ifvg"
                        bull_fvgs.append(f)
                    continue
                new_bear.append(f)
        bear_fvgs = new_bear

        # --- Detect new bull FVG at bar i ---
        if i >= 2:
            gap = lows[i] - highs[i - 2]
            min_sz = a_i * min_size_atr
            if gap > min_sz:
                top = lows[i]
                bot = highs[i - 2]
                ce = (top + bot) / 2.0

                # Displacement check on middle candle (i-1)
                is_disp = 0
                if has_disp and disp_bull[i - 1]:
                    is_disp = 1
                elif not has_disp:
                    ok, _, _ = _is_displacement_candle(
                        opens[i - 1], highs[i - 1], lows[i - 1], closes[i - 1],
                        atr[i - 1] if not np.isnan(atr[i - 1]) else 1e-9, "bull")
                    if ok:
                        is_disp = 1

                bull_fvgs.append({
                    "top": top, "bot": bot, "ce": ce,
                    "width": top - bot, "age": 0,
                    "ce_touched": 0, "ce_rejected": 0,
                    "is_disp": is_disp, "is_ifvg": 0,
                })

        # --- Detect new bear FVG at bar i ---
        if i >= 2:
            gap = lows[i - 2] - highs[i]
            min_sz = a_i * min_size_atr
            if gap > min_sz:
                top = lows[i - 2]
                bot = highs[i]
                ce = (top + bot) / 2.0

                is_disp = 0
                if has_disp and disp_bear[i - 1]:
                    is_disp = 1
                elif not has_disp:
                    ok, _, _ = _is_displacement_candle(
                        opens[i - 1], highs[i - 1], lows[i - 1], closes[i - 1],
                        atr[i - 1] if not np.isnan(atr[i - 1]) else 1e-9, "bear")
                    if ok:
                        is_disp = 1

                bear_fvgs.append({
                    "top": top, "bot": bot, "ce": ce,
                    "width": top - bot, "age": 0,
                    "ce_touched": 0, "ce_rejected": 0,
                    "is_disp": is_disp, "is_ifvg": 0,
                })

        # --- Update state for all FVGs ---
        for f in bull_fvgs:
            f["age"] += 1
            # CE touch check (bull FVG: low touches CE)
            if not f["ce_touched"] and l_i <= f["ce"]:
                f["ce_touched"] = 1
            # CE rejection (touched CE then closed back above)
            if f["ce_touched"] and not f["ce_rejected"] and c_i > f["ce"]:
                f["ce_rejected"] = 1

        for f in bear_fvgs:
            f["age"] += 1
            # CE touch check (bear FVG: high touches CE)
            if not f["ce_touched"] and h_i >= f["ce"]:
                f["ce_touched"] = 1
            # CE rejection (touched CE then closed back below)
            if f["ce_touched"] and not f["ce_rejected"] and c_i < f["ce"]:
                f["ce_rejected"] = 1

        # --- Rank and output top-3 per direction ---
        for direction, fvg_list in [("bull", bull_fvgs), ("bear", bear_fvgs)]:
            out[f"fvg_{direction}_count"][i] = float(len(fvg_list))
            if not fvg_list:
                continue

            out[f"fvg_{direction}_recent_age"][i] = float(fvg_list[-1]["age"])

            # Rank by score
            ranked = []
            for f in fvg_list:
                d_atr = abs(c_i - f["ce"]) / a_i if va else 999.0
                w_atr = f["width"] / a_i if va else 0.0
                score = _fvg_rank_score(d_atr, f["age"], w_atr, f["is_disp"])
                ranked.append((score, d_atr, f))
            ranked.sort(key=lambda x: (-x[0], x[1]))

            for k in range(min(3, len(ranked))):
                _, d_atr, f = ranked[k]
                p = f"fvg_{direction}_{k + 1}"
                top, bot = f["top"], f["bot"]
                w = max(top - bot, 1e-9)

                out[f"{p}_ce"][i] = f["ce"]
                out[f"{p}_dist_to_ce_atr"][i] = (
                    abs(c_i - f["ce"]) / a_i if va else np.nan)

                out[f"{p}_ce_touched"][i] = float(f["ce_touched"])
                out[f"{p}_ce_rejected"][i] = float(f["ce_rejected"])

                # Fill fraction
                if direction == "bull":
                    ff = max(0.0, min(1.0, (top - min(l_i, top)) / w))
                else:
                    ff = max(0.0, min(1.0, (max(h_i, bot) - bot) / w))
                out[f"{p}_fill_fraction"][i] = ff

                # Fully filled
                if direction == "bull":
                    out[f"{p}_fully_filled"][i] = 1.0 if l_i <= bot else 0.0
                else:
                    out[f"{p}_fully_filled"][i] = 1.0 if h_i >= top else 0.0

                out[f"{p}_is_displacement"][i] = float(f["is_disp"])
                out[f"{p}_is_ifvg"][i] = float(f["is_ifvg"])

    return pd.DataFrame(out, index=df.index)


# ---------------------------------------------------------------------------
# 6. compute_ote_dist (D53 Task H -- added 0.705 level)
# ---------------------------------------------------------------------------
def compute_ote_dist(
    df: pd.DataFrame,
    fib_low: float = 0.618,
    fib_high: float = 0.786,
    swing_lookback: int = 20,
) -> pd.DataFrame:
    """
    ATR-normalized distance from close to OTE zone midpoint, plus
    0.705 ICT sweet-spot level.

    OTE zone: fib_low to fib_high retracement of the most recent
    swing range. Uses forward-filled swing prices.

    Requires: swing_high_price, swing_low_price in df.

    Returns DataFrame with columns:
      ote_dist, ote_dist_from_705_atr, ote_at_705
    """
    closes = df["close"].values
    atr = _ensure_atr(df)

    sh_p = df["swing_high_price"].ffill().values
    sl_p = df["swing_low_price"].ffill().values

    swing_range = sh_p - sl_p
    valid = (atr > 0) & ~np.isnan(swing_range) & (swing_range > 0)

    fib_mid = (fib_low + fib_high) / 2
    ote_mid = sh_p - fib_mid * swing_range

    dist = np.where(valid, (closes - ote_mid) / atr, np.nan)

    # pd_position = (close - swing_low) / range, 0..1
    rng_safe = np.where(swing_range > 0, swing_range, np.nan)
    pd_position = (closes - sl_p) / rng_safe

    # 0.705 level: ICT sweet spot within OTE zone
    dist_705 = np.where(valid, np.abs(pd_position - 0.705) / (atr / rng_safe), np.nan)
    at_705 = np.where(valid, (np.abs(pd_position - 0.705) < 0.05).astype(np.float32), np.nan)

    return pd.DataFrame({
        "ote_dist": dist.astype(np.float32),
        "ote_dist_from_705_atr": dist_705.astype(np.float32),
        "ote_at_705": at_705.astype(np.float32),
    }, index=df.index)


# ---------------------------------------------------------------------------
# 7. compute_liq_levels
# ---------------------------------------------------------------------------
def compute_liq_levels(
    df: pd.DataFrame,
    eq_tolerance_atr: float = 0.10,
    lookback: int = 50,
) -> pd.DataFrame:
    """
    Nearest intact liquidity levels above/below close, plus equal
    highs/lows and previous day/week levels.

    Requires: swing_high, swing_low, swing_high_price, swing_low_price in df.

    Returns DataFrame with columns:
      liq_dist_above_pct, liq_dist_below_pct, liq_eq_high, liq_eq_low,
      liq_pdh, liq_pdl, liq_pwh, liq_pwl
    """
    size = len(df)
    closes = df["close"].values
    sh = df["swing_high"].values
    sl = df["swing_low"].values
    sh_p = df["swing_high_price"].values
    sl_p = df["swing_low_price"].values
    atr = _ensure_atr(df)

    liq_above_pct = np.full(size, np.nan)
    liq_below_pct = np.full(size, np.nan)
    eq_high = np.zeros(size, dtype=np.int8)
    eq_low = np.zeros(size, dtype=np.int8)

    intact_highs = []
    intact_lows = []

    for i in range(size):
        if sh[i] == 1 and not np.isnan(sh_p[i]):
            intact_highs.append(sh_p[i])
        if sl[i] == 1 and not np.isnan(sl_p[i]):
            intact_lows.append(sl_p[i])

        c = closes[i]

        above = [p for p in intact_highs if p > c]
        below = [p for p in intact_lows if p < c]
        if above:
            nearest_above = min(above)
            liq_above_pct[i] = (nearest_above - c) / c * 100
            tol = atr[i] * eq_tolerance_atr if not np.isnan(atr[i]) else 0
            if tol > 0 and sum(1 for p in above if abs(p - nearest_above) <= tol) >= 2:
                eq_high[i] = 1
        if below:
            nearest_below = max(below)
            liq_below_pct[i] = (c - nearest_below) / c * 100
            tol = atr[i] * eq_tolerance_atr if not np.isnan(atr[i]) else 0
            if tol > 0 and sum(1 for p in below if abs(p - nearest_below) <= tol) >= 2:
                eq_low[i] = 1

        # Remove swept levels
        intact_highs = [p for p in intact_highs if p > c]
        intact_lows = [p for p in intact_lows if p < c]

        # Cap list size for performance
        if len(intact_highs) > lookback:
            intact_highs = intact_highs[-lookback:]
        if len(intact_lows) > lookback:
            intact_lows = intact_lows[-lookback:]

    # Previous day/week highs and lows (causal: previous period only)
    pdh = np.full(size, np.nan)
    pdl = np.full(size, np.nan)
    pwh = np.full(size, np.nan)
    pwl = np.full(size, np.nan)

    if "bar_start_ts_utc" in df.columns:
        ts = df["bar_start_ts_utc"]
        dates = ts.dt.date
        weeks = ts.dt.to_period("W")

        daily_agg = df.groupby(dates).agg(
            dh=("high", "max"), dl=("low", "min"),
        ).reset_index(names=["_date"])
        daily_agg["pdh"] = daily_agg["dh"].shift(1)
        daily_agg["pdl"] = daily_agg["dl"].shift(1)
        date_map_h = dict(zip(daily_agg["_date"], daily_agg["pdh"]))
        date_map_l = dict(zip(daily_agg["_date"], daily_agg["pdl"]))
        for i in range(size):
            d = dates.iloc[i]
            if d in date_map_h:
                pdh[i] = date_map_h[d]
            if d in date_map_l:
                pdl[i] = date_map_l[d]

        weekly_agg = df.groupby(weeks).agg(
            wh=("high", "max"), wl=("low", "min"),
        ).reset_index(names=["_week"])
        weekly_agg["pwh"] = weekly_agg["wh"].shift(1)
        weekly_agg["pwl"] = weekly_agg["wl"].shift(1)
        week_map_h = dict(zip(weekly_agg["_week"], weekly_agg["pwh"]))
        week_map_l = dict(zip(weekly_agg["_week"], weekly_agg["pwl"]))
        for i in range(size):
            w = weeks.iloc[i]
            if w in week_map_h:
                pwh[i] = week_map_h[w]
            if w in week_map_l:
                pwl[i] = week_map_l[w]

    return pd.DataFrame({
        "liq_dist_above_pct": liq_above_pct,
        "liq_dist_below_pct": liq_below_pct,
        "liq_eq_high": eq_high,
        "liq_eq_low": eq_low,
        "liq_pdh": pdh,
        "liq_pdl": pdl,
        "liq_pwh": pwh,
        "liq_pwl": pwl,
    }, index=df.index)


# ---------------------------------------------------------------------------
# 7b. detect_sweep (D53 Task B)
# ---------------------------------------------------------------------------
def detect_sweep(
    df: pd.DataFrame,
    eq_tolerance_atr: float = 0.10,
    lookback_bars: int = 50,
    m: int = 2,
    validity_bars: int = 6,
) -> pd.DataFrame:
    """
    BSL/SSL sweep detection with EQH/EQL clustering.

    BSL sweep (bearish): wick above EQH, close back below within m bars.
    SSL sweep (bullish): wick below EQL, close back above within m bars.

    Uses confirmed swing highs/lows (int_swing_high/low if available,
    else swing_high/low from compute_swing_points).

    Returns DataFrame with 8 columns:
      sweep_bsl_fired, sweep_bsl_age, sweep_bsl_pen_atr,
      sweep_ssl_fired, sweep_ssl_age, sweep_ssl_pen_atr,
      dist_unswept_bsl_atr, dist_unswept_ssl_atr
    """
    size = len(df)
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    atr = _ensure_atr(df)

    # Use internal swing layer if available, else original
    if "int_swing_high" in df.columns:
        sh_bool = df["int_swing_high"].values
        sh_price = df["int_swing_high_price"].values
        sl_bool = df["int_swing_low"].values
        sl_price = df["int_swing_low_price"].values
    else:
        sh_bool = df["swing_high"].values
        sh_price = df["swing_high_price"].values
        sl_bool = df["swing_low"].values
        sl_price = df["swing_low_price"].values

    bsl_fired = np.zeros(size, dtype=np.float32)
    ssl_fired = np.zeros(size, dtype=np.float32)
    bsl_age = np.full(size, np.nan, dtype=np.float32)
    ssl_age = np.full(size, np.nan, dtype=np.float32)
    bsl_pen = np.zeros(size, dtype=np.float32)
    ssl_pen = np.zeros(size, dtype=np.float32)
    dist_bsl = np.full(size, np.nan, dtype=np.float32)
    dist_ssl = np.full(size, np.nan, dtype=np.float32)

    pending_bsl = None  # (level, breach_bar, penetration)
    pending_ssl = None
    last_bsl = None
    last_ssl = None

    def _cluster_levels(prices, tol):
        """Cluster nearby prices; return (mean, count) for clusters with >= 2."""
        if not prices:
            return []
        ps = sorted(prices)
        clusters = []
        cur = [ps[0]]
        for p in ps[1:]:
            if abs(p - cur[-1]) <= tol:
                cur.append(p)
            else:
                clusters.append(cur)
                cur = [p]
        clusters.append(cur)
        return [(sum(c) / len(c), len(c)) for c in clusters if len(c) >= 2]

    for t in range(size):
        a_t = atr[t]
        va = not (np.isnan(a_t) or a_t <= 0)
        if not va:
            a_t = 1e-9
        tol = eq_tolerance_atr * a_t
        c_t = closes[t]

        # Gather recent swing prices
        lo_idx = max(0, t - lookback_bars)
        hi_prices = []
        lo_prices = []
        for j in range(lo_idx, t):
            if sh_bool[j] and not np.isnan(sh_price[j]):
                hi_prices.append(sh_price[j])
            if sl_bool[j] and not np.isnan(sl_price[j]):
                lo_prices.append(sl_price[j])

        eqh = _cluster_levels(hi_prices, tol)
        eql = _cluster_levels(lo_prices, tol)

        # Nearest unswept levels
        bsl_lvls = [lvl for (lvl, _) in eqh if lvl > c_t]
        ssl_lvls = [lvl for (lvl, _) in eql if lvl < c_t]
        if bsl_lvls and va:
            dist_bsl[t] = (min(bsl_lvls) - c_t) / a_t
        if ssl_lvls and va:
            dist_ssl[t] = (c_t - max(ssl_lvls)) / a_t

        # Confirm pending BSL (wick above EQH, close back below)
        if pending_bsl is not None:
            lvl, breach_bar, pen = pending_bsl
            if t - breach_bar > m:
                pending_bsl = None
            elif closes[t] < lvl:
                bsl_fired[t] = 1.0
                bsl_pen[t] = pen
                last_bsl = t
                pending_bsl = None

        # Confirm pending SSL (wick below EQL, close back above)
        if pending_ssl is not None:
            lvl, breach_bar, pen = pending_ssl
            if t - breach_bar > m:
                pending_ssl = None
            elif closes[t] > lvl:
                ssl_fired[t] = 1.0
                ssl_pen[t] = pen
                last_ssl = t
                pending_ssl = None

        # New BSL breach
        nearest_eqh = min(bsl_lvls, default=None)
        if nearest_eqh is not None and pending_bsl is None and highs[t] > nearest_eqh:
            pending_bsl = (nearest_eqh, t,
                           (highs[t] - nearest_eqh) / a_t if va else 0.0)

        # New SSL breach
        nearest_eql = max(ssl_lvls, default=None) if ssl_lvls else None
        if nearest_eql is not None and pending_ssl is None and lows[t] < nearest_eql:
            pending_ssl = (nearest_eql, t,
                           (nearest_eql - lows[t]) / a_t if va else 0.0)

        # Freshness ages
        if last_bsl is not None:
            age = t - last_bsl
            if 0 <= age <= validity_bars:
                bsl_age[t] = float(age)
        if last_ssl is not None:
            age = t - last_ssl
            if 0 <= age <= validity_bars:
                ssl_age[t] = float(age)

    return pd.DataFrame({
        "sweep_bsl_fired": bsl_fired,
        "sweep_bsl_age": bsl_age,
        "sweep_bsl_pen_atr": bsl_pen,
        "sweep_ssl_fired": ssl_fired,
        "sweep_ssl_age": ssl_age,
        "sweep_ssl_pen_atr": ssl_pen,
        "dist_unswept_bsl_atr": dist_bsl,
        "dist_unswept_ssl_atr": dist_ssl,
    }, index=df.index)


# ---------------------------------------------------------------------------
# 7c. detect_sweep_sequence (D53 Task B -- composite)
# ---------------------------------------------------------------------------
def detect_sweep_sequence(
    df: pd.DataFrame,
    k: int = 6,
    N: int = 12,
) -> pd.DataFrame:
    """
    Sweep sequence composite: sweep -> displacement -> FVG within N bars.

    Bull: SSL sweep -> bull displacement within k bars -> bull FVG within k bars.
    Bear: BSL sweep -> bear displacement -> bear FVG. Fires at FVG bar.

    Requires: sweep_ssl_fired, sweep_bsl_fired, displacement_bull,
              displacement_bear, fvg_bull_in_zone, fvg_bear_in_zone in df.

    Returns DataFrame with 4 columns:
      sweep_seq_bull_complete, sweep_seq_bull_age,
      sweep_seq_bear_complete, sweep_seq_bear_age
    """
    size = len(df)
    ssl_fired = df["sweep_ssl_fired"].values if "sweep_ssl_fired" in df.columns else np.zeros(size)
    bsl_fired = df["sweep_bsl_fired"].values if "sweep_bsl_fired" in df.columns else np.zeros(size)
    disp_bull = df["displacement_bull"].values if "displacement_bull" in df.columns else np.zeros(size)
    disp_bear = df["displacement_bear"].values if "displacement_bear" in df.columns else np.zeros(size)

    # Use FVG formation detection: check if a new FVG formed at this bar
    fvg_bull_count = df["fvg_bull_count"].values if "fvg_bull_count" in df.columns else np.zeros(size)
    fvg_bear_count = df["fvg_bear_count"].values if "fvg_bear_count" in df.columns else np.zeros(size)

    seq_bull = np.zeros(size, dtype=np.float32)
    seq_bull_age = np.full(size, np.nan, dtype=np.float32)
    seq_bear = np.zeros(size, dtype=np.float32)
    seq_bear_age = np.full(size, np.nan, dtype=np.float32)

    last_bull_seq = -1
    last_bear_seq = -1

    for t in range(size):
        lo = max(0, t - N)

        # Bull sequence: SSL sweep -> bull disp -> bull FVG
        # Check if bull FVG formed at this bar
        fvg_bull_now = (t >= 1 and fvg_bull_count[t] > fvg_bull_count[t - 1])
        if fvg_bull_now:
            # Check for bull displacement within k bars before
            disp_lo = max(0, t - k)
            has_disp = any(disp_bull[j] for j in range(disp_lo, t + 1))
            if has_disp:
                # Check for SSL sweep within N bars before
                sweep_lo = max(0, t - N)
                has_sweep = any(ssl_fired[j] for j in range(sweep_lo, t + 1))
                if has_sweep:
                    seq_bull[t] = 1.0
                    last_bull_seq = t

        # Bear sequence: BSL sweep -> bear disp -> bear FVG
        fvg_bear_now = (t >= 1 and fvg_bear_count[t] > fvg_bear_count[t - 1])
        if fvg_bear_now:
            disp_lo = max(0, t - k)
            has_disp = any(disp_bear[j] for j in range(disp_lo, t + 1))
            if has_disp:
                sweep_lo = max(0, t - N)
                has_sweep = any(bsl_fired[j] for j in range(sweep_lo, t + 1))
                if has_sweep:
                    seq_bear[t] = 1.0
                    last_bear_seq = t

        # Ages
        if last_bull_seq >= 0:
            age = t - last_bull_seq
            if age <= N:
                seq_bull_age[t] = float(age)
        if last_bear_seq >= 0:
            age = t - last_bear_seq
            if age <= N:
                seq_bear_age[t] = float(age)

    return pd.DataFrame({
        "sweep_seq_bull_complete": seq_bull,
        "sweep_seq_bull_age": seq_bull_age,
        "sweep_seq_bear_complete": seq_bear,
        "sweep_seq_bear_age": seq_bear_age,
    }, index=df.index)


# ---------------------------------------------------------------------------
# 8. compute_premium_discount (D53 Task G -- continuous 0-1)
# ---------------------------------------------------------------------------
def compute_premium_discount(
    df: pd.DataFrame,
    swing_lookback: int = 96,
) -> pd.DataFrame:
    """
    Continuous premium/discount position using external swing layer prices.

    pd_position = (close - swing_low) / (swing_high - swing_low), clipped to [0, 1].
    0 = at swing low (deep discount), 1 = at swing high (deep premium).

    Uses ext_swing_high_price / ext_swing_low_price if available (from
    compute_swing_dual_layer), otherwise falls back to swing_high_price /
    swing_low_price.

    Returns DataFrame with 7 columns:
      pd_position_5m, pd_dist_from_eq, in_discount, in_deep_discount,
      in_deep_premium, in_ote_bull, in_ote_bear
    """
    # Use external swing prices if available, else fall back
    if "ext_swing_high_price" in df.columns:
        sh_p = df["ext_swing_high_price"].ffill().values
        sl_p = df["ext_swing_low_price"].ffill().values
    else:
        sh_p = df["swing_high_price"].ffill().values
        sl_p = df["swing_low_price"].ffill().values

    closes = df["close"].values
    rng = sh_p - sl_p
    # Avoid division by zero
    rng_safe = np.where(rng > 0, rng, np.nan)

    pd_pos = np.clip((closes - sl_p) / rng_safe, 0.0, 1.0).astype(np.float32)

    return pd.DataFrame({
        "pd_position_5m": pd_pos,
        "pd_dist_from_eq": np.abs(pd_pos - 0.5).astype(np.float32),
        "in_discount": (pd_pos < 0.5).astype(np.float32),
        "in_deep_discount": (pd_pos <= 0.25).astype(np.float32),
        "in_deep_premium": (pd_pos >= 0.75).astype(np.float32),
        "in_ote_bull": ((pd_pos >= 0.214) & (pd_pos <= 0.382)).astype(np.float32),
        "in_ote_bear": ((pd_pos >= 0.618) & (pd_pos <= 0.786)).astype(np.float32),
    }, index=df.index)


# ---------------------------------------------------------------------------
# 9. compute_cisd (D53 Task E -- corrected algorithm)
# ---------------------------------------------------------------------------
def compute_cisd(
    df: pd.DataFrame,
    L: int = 20,
    min_run: int = 2,
    age_cap: int = 20,
) -> pd.DataFrame:
    """
    Change in State of Delivery (CISD) -- corrected algorithm (D53).

    Bullish CISD:
      1. Find most recent run of >= min_run consecutive bearish candles in last L bars.
      2. delivery_open = open of FIRST candle in that run.
      3. Fires when: close[t] > delivery_open AND close[t-1] <= delivery_open.

    Bearish CISD: mirror with bullish candle runs.

    cisd_with_sweep: also requires sweep within 6 bars.

    Returns DataFrame with 6 columns:
      cisd_bull, cisd_bear, cisd_bull_age, cisd_bear_age,
      cisd_bull_with_sweep, cisd_bear_with_sweep
    """
    size = len(df)
    opens = df["open"].values
    closes = df["close"].values

    # Optional sweep data
    has_ssl = "sweep_ssl_age" in df.columns
    has_bsl = "sweep_bsl_age" in df.columns
    ssl_age_arr = df["sweep_ssl_age"].values if has_ssl else None
    bsl_age_arr = df["sweep_bsl_age"].values if has_bsl else None

    cb = np.zeros(size, dtype=np.float32)
    cr = np.zeros(size, dtype=np.float32)
    cb_age = np.full(size, np.nan, dtype=np.float32)
    cr_age = np.full(size, np.nan, dtype=np.float32)
    cb_ws = np.zeros(size, dtype=np.float32)
    cr_ws = np.zeros(size, dtype=np.float32)

    last_cb = -1
    last_cr = -1

    for t in range(1, size):
        # --- Bullish CISD: find bearish candle run ---
        run = 0
        start = None
        for j in range(t - 1, max(-1, t - L), -1):
            if closes[j] < opens[j]:  # bearish candle
                run += 1
                start = j
            else:
                if run >= min_run:
                    break
                run = 0
                start = None
        if start is not None and run >= min_run:
            level = opens[start]
            if closes[t] > level and closes[t - 1] <= level:
                cb[t] = 1.0
                last_cb = t
                # Check for SSL sweep within 6 bars
                if has_ssl and ssl_age_arr is not None:
                    sa = ssl_age_arr[t]
                    if not np.isnan(sa) and sa <= 6:
                        cb_ws[t] = 1.0

        # --- Bearish CISD: find bullish candle run ---
        run = 0
        start = None
        for j in range(t - 1, max(-1, t - L), -1):
            if closes[j] > opens[j]:  # bullish candle
                run += 1
                start = j
            else:
                if run >= min_run:
                    break
                run = 0
                start = None
        if start is not None and run >= min_run:
            level = opens[start]
            if closes[t] < level and closes[t - 1] >= level:
                cr[t] = 1.0
                last_cr = t
                if has_bsl and bsl_age_arr is not None:
                    ba = bsl_age_arr[t]
                    if not np.isnan(ba) and ba <= 6:
                        cr_ws[t] = 1.0

        # Ages
        if last_cb >= 0:
            age = t - last_cb
            if age <= age_cap:
                cb_age[t] = float(age)
        if last_cr >= 0:
            age = t - last_cr
            if age <= age_cap:
                cr_age[t] = float(age)

    return pd.DataFrame({
        "cisd_bull": cb,
        "cisd_bear": cr,
        "cisd_bull_age": cb_age,
        "cisd_bear_age": cr_age,
        "cisd_bull_with_sweep": cb_ws,
        "cisd_bear_with_sweep": cr_ws,
    }, index=df.index)


# ---------------------------------------------------------------------------
# 10. compute_ob_quality
# ---------------------------------------------------------------------------
def compute_ob_quality(
    df: pd.DataFrame,
    lookback: int = 200,
    vol_window: int = 20,
) -> pd.DataFrame:
    """
    Order block quality score combining recency, displacement, and volume.

    ob_quality_score = recency_weight * displacement_strength * volume_surge

    Where:
      recency_weight      = 1 / (ob_age + 1)
      displacement_strength = abs(close - open) / ATR at OB formation bar
      volume_surge        = volume at OB bar / rolling_mean_volume(vol_window)

    These values are fixed at OB formation time (the formation bar's body
    size and volume). Only recency_weight changes as the OB ages.

    Requires: bos_close column in df (from compute_swing_points).
    Uses: open, close, high, low, volume_base (or volume), ict_atr_14.

    Returns DataFrame with columns:
      ob_bull_quality, ob_bear_quality
    """
    size = len(df)
    opens = df["open"].values
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    bos = df["bos_close"].values
    atr = _ensure_atr(df)

    # Volume: prefer volume_base, fall back to volume
    if "volume_base" in df.columns:
        vol = df["volume_base"].values.astype(np.float64)
    elif "volume" in df.columns:
        vol = df["volume"].values.astype(np.float64)
    else:
        vol = np.ones(size, dtype=np.float64)

    # Rolling mean volume (causal: expanding then rolling)
    vol_rm = pd.Series(vol).rolling(vol_window, min_periods=1).mean().values

    # Bull OB tracking
    bull_quality = np.full(size, np.nan)
    bull_disp = 0.0
    bull_vsurge = 0.0
    bull_active = False
    bull_age = 0

    # Bear OB tracking
    bear_quality = np.full(size, np.nan)
    bear_disp = 0.0
    bear_vsurge = 0.0
    bear_active = False
    bear_age = 0

    for i in range(size):
        # --- Bull OB detection (same logic as detect_ob_bull) ---
        if bos[i] == 1:
            found = False
            search_start = max(0, i - lookback)
            for j in range(i - 1, search_start - 1, -1):
                if closes[j] < opens[j]:  # bearish candle = bull OB
                    body = abs(closes[j] - opens[j])
                    a = atr[j] if not np.isnan(atr[j]) and atr[j] > 0 else 1.0
                    bull_disp = body / a
                    rm = vol_rm[j] if vol_rm[j] > 0 else 1.0
                    bull_vsurge = vol[j] / rm
                    bull_active = True
                    bull_age = 0
                    found = True
                    break
            if not found:
                bull_active = False

        if bull_active:
            recency = 1.0 / (bull_age + 1)
            bull_quality[i] = recency * bull_disp * bull_vsurge
            bull_age += 1

        # --- Bear OB detection (same logic as detect_ob_bear) ---
        if bos[i] == -1:
            found = False
            search_start = max(0, i - lookback)
            for j in range(i - 1, search_start - 1, -1):
                if closes[j] > opens[j]:  # bullish candle = bear OB
                    body = abs(closes[j] - opens[j])
                    a = atr[j] if not np.isnan(atr[j]) and atr[j] > 0 else 1.0
                    bear_disp = body / a
                    rm = vol_rm[j] if vol_rm[j] > 0 else 1.0
                    bear_vsurge = vol[j] / rm
                    bear_active = True
                    bear_age = 0
                    found = True
                    break
            if not found:
                bear_active = False

        if bear_active:
            recency = 1.0 / (bear_age + 1)
            bear_quality[i] = recency * bear_disp * bear_vsurge
            bear_age += 1

    return pd.DataFrame({
        "ob_bull_quality": bull_quality,
        "ob_bear_quality": bear_quality,
    }, index=df.index)


# ---------------------------------------------------------------------------
# 11. detect_breaker_blocks
# ---------------------------------------------------------------------------
def detect_breaker_blocks(
    df: pd.DataFrame,
    lookback: int = 200,
    age_cap: int = 200,
) -> pd.DataFrame:
    """
    Breaker block detection: OBs that get mitigated then flip direction.

    Bull breaker: a bear OB gets mitigated (close rises above ob_top),
    the zone flips to act as support. This is a bullish signal.

    Bear breaker: a bull OB gets mitigated (close falls below ob_bot),
    the zone flips to act as resistance. This is a bearish signal.

    A breaker expires after age_cap bars or when price breaks through
    the breaker zone itself (support/resistance fails).

    Requires: bos_close column in df (from compute_swing_points).

    Returns DataFrame with columns:
      breaker_bull_age, breaker_bull_dist, breaker_bull_in_zone,
      breaker_bear_age, breaker_bear_dist, breaker_bear_in_zone
    """
    size = len(df)
    opens = df["open"].values
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    bos = df["bos_close"].values
    atr = _ensure_atr(df)

    # Output arrays
    out_bull_age = np.full(size, np.nan)
    out_bull_dist = np.full(size, np.nan)
    out_bull_zone = np.zeros(size, dtype=np.int8)
    out_bear_age = np.full(size, np.nan)
    out_bear_dist = np.full(size, np.nan)
    out_bear_zone = np.zeros(size, dtype=np.int8)

    # --- Active OB state (tracks latest OB of each direction) ---
    bull_ob_top = np.nan
    bull_ob_bot = np.nan
    bull_ob_active = False

    bear_ob_top = np.nan
    bear_ob_bot = np.nan
    bear_ob_active = False

    # --- Active breaker state ---
    bkr_bull_top = np.nan
    bkr_bull_bot = np.nan
    bkr_bull_active = False
    bkr_bull_age = 0

    bkr_bear_top = np.nan
    bkr_bear_bot = np.nan
    bkr_bear_active = False
    bkr_bear_age = 0

    for i in range(size):
        # === 1. Detect new bull OB at bullish BOS ===
        if bos[i] == 1:
            search_start = max(0, i - lookback)
            for j in range(i - 1, search_start - 1, -1):
                if closes[j] < opens[j]:  # bearish candle = bull OB
                    bull_ob_top = opens[j]   # body top (open of red candle)
                    bull_ob_bot = closes[j]  # body bot (close of red candle)
                    bull_ob_active = True
                    break

        # === 2. Detect new bear OB at bearish BOS ===
        if bos[i] == -1:
            search_start = max(0, i - lookback)
            for j in range(i - 1, search_start - 1, -1):
                if closes[j] > opens[j]:  # bullish candle = bear OB
                    bear_ob_top = closes[j]  # body top (close of green candle)
                    bear_ob_bot = opens[j]   # body bot (open of green candle)
                    bear_ob_active = True
                    break

        # === 3. Check bull OB mitigation -> creates bear breaker ===
        # Bull OB mitigated when close drops below the OB zone
        if bull_ob_active and closes[i] < bull_ob_bot:
            bkr_bear_top = bull_ob_top
            bkr_bear_bot = bull_ob_bot
            bkr_bear_active = True
            bkr_bear_age = 0
            bull_ob_active = False

        # === 4. Check bear OB mitigation -> creates bull breaker ===
        # Bear OB mitigated when close rises above the OB zone
        if bear_ob_active and closes[i] > bear_ob_top:
            bkr_bull_top = bear_ob_top
            bkr_bull_bot = bear_ob_bot
            bkr_bull_active = True
            bkr_bull_age = 0
            bear_ob_active = False

        # === 5. Expire / invalidate breakers ===
        # Bull breaker (support) fails when close drops below its bot
        if bkr_bull_active:
            if bkr_bull_age >= age_cap or closes[i] < bkr_bull_bot:
                bkr_bull_active = False

        # Bear breaker (resistance) fails when close rises above its top
        if bkr_bear_active:
            if bkr_bear_age >= age_cap or closes[i] > bkr_bear_top:
                bkr_bear_active = False

        # === 6. Output breaker features ===
        if bkr_bull_active:
            out_bull_age[i] = bkr_bull_age
            mid = (bkr_bull_top + bkr_bull_bot) / 2
            a = atr[i]
            if not np.isnan(a) and a > 0:
                out_bull_dist[i] = (closes[i] - mid) / a
            if lows[i] <= bkr_bull_top and highs[i] >= bkr_bull_bot:
                out_bull_zone[i] = 1
            bkr_bull_age += 1

        if bkr_bear_active:
            out_bear_age[i] = bkr_bear_age
            mid = (bkr_bear_top + bkr_bear_bot) / 2
            a = atr[i]
            if not np.isnan(a) and a > 0:
                out_bear_dist[i] = (mid - closes[i]) / a
            if lows[i] <= bkr_bear_top and highs[i] >= bkr_bear_bot:
                out_bear_zone[i] = 1
            bkr_bear_age += 1

    return pd.DataFrame({
        "breaker_bull_age": out_bull_age,
        "breaker_bull_dist": out_bull_dist,
        "breaker_bull_in_zone": out_bull_zone,
        "breaker_bear_age": out_bear_age,
        "breaker_bear_dist": out_bear_dist,
        "breaker_bear_in_zone": out_bear_zone,
    }, index=df.index)


# ---------------------------------------------------------------------------
# 12. detect_mss (D53 -- Market Structure Shift)
# ---------------------------------------------------------------------------
MSS_K = 3  # displacement window (bars)


def detect_mss(
    df: pd.DataFrame,
    mss_k: int = 3,
    age_cap: int = 24,
) -> pd.DataFrame:
    """
    Market Structure Shift: CHoCH confirmed by displacement within k bars.

    MSS_bull = int_choch_bull AND any(displacement_bull within mss_k bars).
    MSS_bear = int_choch_bear AND any(displacement_bear within mss_k bars).
    mss_with_sweep: also requires sweep within 10 bars.

    Requires: int_choch_bull, int_choch_bear, displacement_bull,
              displacement_bear columns in df.

    Returns DataFrame with 7 columns:
      mss_bull_fired, mss_bear_fired, mss_bull_age, mss_bear_age,
      mss_strength, mss_bull_with_sweep, mss_bear_with_sweep
    """
    size = len(df)

    # CHoCH from internal swing layer (or original swings)
    if "int_choch_bull" in df.columns:
        choch_bull = df["int_choch_bull"].values
        choch_bear = df["int_choch_bear"].values
    else:
        # Fallback: use choch_close from compute_swing_points
        choch = df["choch_close"].values if "choch_close" in df.columns else np.zeros(size)
        choch_bull = (choch == 1).astype(np.float64)
        choch_bear = (choch == -1).astype(np.float64)

    # Displacement
    disp_bull = df["displacement_bull"].values if "displacement_bull" in df.columns else np.zeros(size)
    disp_bear = df["displacement_bear"].values if "displacement_bear" in df.columns else np.zeros(size)
    disp_str = np.zeros(size)
    if "displacement_bull_strength" in df.columns:
        bs = df["displacement_bull_strength"].values
        rs = df["displacement_bear_strength"].values
        disp_str = np.where(~np.isnan(bs), bs, np.where(~np.isnan(rs), rs, 0.0))

    # Sweep ages (optional)
    has_ssl = "sweep_ssl_age" in df.columns
    has_bsl = "sweep_bsl_age" in df.columns
    ssl_age_arr = df["sweep_ssl_age"].values if has_ssl else None
    bsl_age_arr = df["sweep_bsl_age"].values if has_bsl else None

    mb = np.zeros(size, dtype=np.float32)
    mr = np.zeros(size, dtype=np.float32)
    mb_age = np.full(size, np.nan, dtype=np.float32)
    mr_age = np.full(size, np.nan, dtype=np.float32)
    m_str = np.zeros(size, dtype=np.float32)
    mb_ws = np.zeros(size, dtype=np.float32)
    mr_ws = np.zeros(size, dtype=np.float32)

    last_mb = -1
    last_mr = -1

    for t in range(size):
        lo = max(0, t - mss_k)

        # Bull MSS: CHoCH bull + displacement bull within k bars
        if choch_bull[t]:
            has_disp = False
            max_str = 0.0
            for j in range(lo, t + 1):
                if disp_bull[j]:
                    has_disp = True
                    if disp_str[j] > max_str:
                        max_str = disp_str[j]
            if has_disp:
                mb[t] = 1.0
                m_str[t] = max_str
                last_mb = t
                # With sweep check
                if has_ssl and ssl_age_arr is not None:
                    sa = ssl_age_arr[t]
                    if not np.isnan(sa) and sa <= 10:
                        mb_ws[t] = 1.0

        # Bear MSS: CHoCH bear + displacement bear within k bars
        if choch_bear[t]:
            has_disp = False
            max_str = 0.0
            for j in range(lo, t + 1):
                if disp_bear[j]:
                    has_disp = True
                    if disp_str[j] > max_str:
                        max_str = disp_str[j]
            if has_disp:
                mr[t] = 1.0
                m_str[t] = max_str
                last_mr = t
                if has_bsl and bsl_age_arr is not None:
                    ba = bsl_age_arr[t]
                    if not np.isnan(ba) and ba <= 10:
                        mr_ws[t] = 1.0

        # Ages
        if last_mb >= 0:
            age = t - last_mb
            if age <= age_cap:
                mb_age[t] = float(age)
        if last_mr >= 0:
            age = t - last_mr
            if age <= age_cap:
                mr_age[t] = float(age)

    return pd.DataFrame({
        "mss_bull_fired": mb,
        "mss_bear_fired": mr,
        "mss_bull_age": mb_age,
        "mss_bear_age": mr_age,
        "mss_strength": m_str,
        "mss_bull_with_sweep": mb_ws,
        "mss_bear_with_sweep": mr_ws,
    }, index=df.index)
