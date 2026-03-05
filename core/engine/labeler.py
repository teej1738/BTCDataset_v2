"""
labeler.py -- Dynamic triple-barrier labeler
D46a: Dynamic Labeler + Fill Model

Computes triple-barrier labels for any combination of:
- Direction: long or short
- Stop type: swing_low, swing_high, atr, fixed_pct
- Entry type: market, limit_ob_mid, limit_fvg_edge, limit_ce
- Target R-multiple: any float (e.g. 1.0, 2.0, 3.0)
- Max hold horizon: any int (bars)

Labels are forward-looking by design (for labeling only).
Entry price and stop use only information available at signal bar T.
The label tells us what HAPPENED after bar T, not what WILL happen.

Public API:
  compute_labels(df, label_config, signal_mask=None) -> np.ndarray
    Returns array of {0, 1, NaN} same length as df.
    1 = target hit before stop within horizon
    0 = stop hit first or timeout
    NaN = invalid (no fill, bad stop, insufficient data, masked out)
"""

import numpy as np

from core.engine.fill_model import compute_entry_price

# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "direction": "long",
    "target_r": 2.0,
    "stop_type": "atr",
    "stop_atr_mult": 1.5,
    "stop_pct": 0.02,
    "max_bars": 48,
    "entry_delay": 0,
    "entry_type": "market",
    "fill_timeout": 12,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def compute_labels(df, label_config, signal_mask=None):
    """Compute dynamic triple-barrier labels.

    Parameters
    ----------
    df : DataFrame
        Must contain: close, high, low, ict_atr_14.
        For swing stops: ict_swing_low_price, ict_swing_high_price.
        For limit entries: relevant OB/FVG/swing columns.
    label_config : dict
        See DEFAULT_CONFIG for schema.
    signal_mask : np.ndarray or None
        Boolean array. NaN label for bars where mask is False.
        If None, label every bar.

    Returns
    -------
    np.ndarray of float64, length = len(df)
        Values: 1.0 (target hit), 0.0 (stop/timeout), NaN (invalid/masked).
    """
    cfg = {**DEFAULT_CONFIG, **label_config}

    entry_type = cfg["entry_type"]
    entry_delay = cfg["entry_delay"]

    if entry_type == "market" and entry_delay == 0:
        labels = _label_vectorized(df, cfg)
    elif entry_type == "market" and entry_delay > 0:
        labels = _label_vectorized(df, cfg)
    else:
        # Limit entries: per-bar processing
        labels = _label_perbar(df, cfg, signal_mask)
        # signal_mask already applied inside _label_perbar
        return labels

    if signal_mask is not None:
        labels[~signal_mask] = np.nan

    return labels


# ---------------------------------------------------------------------------
# Vectorized path (market entries)
# ---------------------------------------------------------------------------
def _label_vectorized(df, cfg):
    """Fully vectorized labeling for market entries."""
    direction = cfg["direction"]
    target_r = cfg["target_r"]
    max_bars = cfg["max_bars"]
    entry_delay = cfg["entry_delay"]

    close = df["close"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    n = len(df)

    # Number of bars we can label
    valid_n = n - entry_delay - max_bars
    if valid_n <= 0:
        return np.full(n, np.nan)

    # Entry prices: close at bar i + entry_delay
    ed = entry_delay
    entry_arr = close[ed: ed + valid_n].copy()

    # Stop levels (computed from signal bar i, not entry bar)
    stop_arr = _compute_stop_array(df, cfg, valid_n)

    # Risk and target
    if direction == "long":
        risk_arr = entry_arr - stop_arr
    else:
        risk_arr = stop_arr - entry_arr

    # Invalid: risk <= 0 or NaN
    invalid = np.isnan(risk_arr) | (risk_arr <= 0) | np.isnan(entry_arr)

    if direction == "long":
        target_arr = entry_arr + target_r * risk_arr
    else:
        target_arr = entry_arr - target_r * risk_arr

    # Build forward windows
    # For bar i, entry at i+ed, forward window = bars [i+ed+1 .. i+ed+max_bars]
    fw_start = ed + 1
    high_fwd = np.lib.stride_tricks.sliding_window_view(
        high[fw_start:], max_bars
    )[:valid_n]
    low_fwd = np.lib.stride_tricks.sliding_window_view(
        low[fw_start:], max_bars
    )[:valid_n]

    # TP and SL detection
    if direction == "long":
        tp_mask = high_fwd >= target_arr[:, np.newaxis]
        sl_mask = low_fwd <= stop_arr[:, np.newaxis]
    else:
        tp_mask = low_fwd <= target_arr[:, np.newaxis]
        sl_mask = high_fwd >= stop_arr[:, np.newaxis]

    first_tp = _first_true(tp_mask)
    first_sl = _first_true(sl_mask)

    # TP before SL -> 1 (win). Tie: SL wins (conservative, matches legacy).
    result = (first_tp < first_sl).astype(np.float64)
    result[invalid] = np.nan

    labels = np.full(n, np.nan)
    labels[:valid_n] = result
    return labels


# ---------------------------------------------------------------------------
# Per-bar path (limit entries)
# ---------------------------------------------------------------------------
def _label_perbar(df, cfg, signal_mask=None):
    """Per-bar labeling for limit entries."""
    direction = cfg["direction"]
    target_r = cfg["target_r"]
    max_bars = cfg["max_bars"]
    entry_delay = cfg["entry_delay"]
    entry_type = cfg["entry_type"]
    fill_timeout = cfg["fill_timeout"]

    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    n = len(df)
    labels = np.full(n, np.nan)

    # Determine which bars to process
    if signal_mask is not None:
        bar_indices = np.where(signal_mask)[0]
    else:
        bar_indices = np.arange(n)

    for i in bar_indices:
        # Get entry price via fill model
        sig_bar = i + entry_delay
        if sig_bar >= n:
            continue

        price, fill_bar = compute_entry_price(
            df, sig_bar, entry_type, fill_timeout, direction
        )
        if np.isnan(price) or fill_bar < 0:
            continue  # no fill -> NaN (already default)

        # Compute stop from signal bar i's data
        stop = _compute_stop_single(df, i, price, cfg)
        if np.isnan(stop):
            continue

        # Compute risk and target
        if direction == "long":
            risk = price - stop
        else:
            risk = stop - price

        if risk <= 0:
            continue

        if direction == "long":
            target = price + target_r * risk
        else:
            target = price - target_r * risk

        # Scan forward from fill_bar+1 for TP/SL
        scan_end = min(fill_bar + 1 + max_bars, n)
        first_tp = scan_end  # sentinel: not found
        first_sl = scan_end

        for j in range(fill_bar + 1, scan_end):
            if direction == "long":
                if high[j] >= target and first_tp == scan_end:
                    first_tp = j
                if low[j] <= stop and first_sl == scan_end:
                    first_sl = j
            else:
                if low[j] <= target and first_tp == scan_end:
                    first_tp = j
                if high[j] >= stop and first_sl == scan_end:
                    first_sl = j

            # Early exit if both found
            if first_tp < scan_end and first_sl < scan_end:
                break

        # Determine label
        if first_tp < first_sl:
            labels[i] = 1.0
        elif first_sl <= first_tp and (first_sl < scan_end or first_tp < scan_end):
            labels[i] = 0.0
        else:
            # Neither hit within horizon -> timeout = 0 (loss)
            labels[i] = 0.0

    return labels


# ---------------------------------------------------------------------------
# Stop computation
# ---------------------------------------------------------------------------
def _compute_stop_array(df, cfg, valid_n):
    """Compute stop level array for the first valid_n bars (vectorized)."""
    stop_type = cfg["stop_type"]
    direction = cfg["direction"]

    if stop_type == "atr":
        atr = df["ict_atr_14"].values[:valid_n].astype(np.float64)
        mult = cfg.get("stop_atr_mult", 1.5)
        entry_delay = cfg.get("entry_delay", 0)
        close = df["close"].values[entry_delay: entry_delay + valid_n].astype(
            np.float64
        )
        if direction == "long":
            return close - atr * mult
        else:
            return close + atr * mult

    elif stop_type == "swing_low":
        # Forward-fill swing low price for a running stop level
        swing = df["ict_swing_low_price"].ffill().values[:valid_n].astype(
            np.float64
        )
        return swing

    elif stop_type == "swing_high":
        swing = df["ict_swing_high_price"].ffill().values[:valid_n].astype(
            np.float64
        )
        return swing

    elif stop_type == "fixed_pct":
        pct = cfg.get("stop_pct", 0.02)
        entry_delay = cfg.get("entry_delay", 0)
        close = df["close"].values[entry_delay: entry_delay + valid_n].astype(
            np.float64
        )
        if direction == "long":
            return close * (1 - pct)
        else:
            return close * (1 + pct)

    else:
        raise ValueError(f"Unknown stop_type: {stop_type}")


def _compute_stop_single(df, i, entry_price, cfg):
    """Compute stop level for a single bar (used in per-bar path)."""
    stop_type = cfg["stop_type"]
    direction = cfg["direction"]

    if stop_type == "atr":
        atr = df["ict_atr_14"].iloc[i]
        if np.isnan(atr):
            return np.nan
        mult = cfg.get("stop_atr_mult", 1.5)
        if direction == "long":
            return entry_price - atr * mult
        else:
            return entry_price + atr * mult

    elif stop_type == "swing_low":
        # Last known swing low at or before bar i
        vals = df["ict_swing_low_price"].values
        for j in range(i, -1, -1):
            if not np.isnan(vals[j]):
                return vals[j]
        return np.nan

    elif stop_type == "swing_high":
        vals = df["ict_swing_high_price"].values
        for j in range(i, -1, -1):
            if not np.isnan(vals[j]):
                return vals[j]
        return np.nan

    elif stop_type == "fixed_pct":
        pct = cfg.get("stop_pct", 0.02)
        if direction == "long":
            return entry_price * (1 - pct)
        else:
            return entry_price * (1 + pct)

    else:
        raise ValueError(f"Unknown stop_type: {stop_type}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _first_true(mask_2d):
    """For each row, return column index of first True.

    Returns ncols if no True found (sentinel value).
    Uses np.argmax which returns 0 for all-False rows,
    so we distinguish with an any() check.
    """
    ncols = mask_2d.shape[1]
    first = np.argmax(mask_2d, axis=1)
    any_true = mask_2d.any(axis=1)
    first[~any_true] = ncols
    return first
