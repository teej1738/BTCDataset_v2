"""
fill_model.py -- Fill price computation for limit and market entries
D46a: Dynamic Labeler + Fill Model

Computes entry prices for different order types:
- market:        close of signal bar (immediate fill)
- limit_ob_mid:  midpoint of most recent order block in trade direction
- limit_fvg_edge: edge of nearest active FVG in trade direction
- limit_ce:      consequent encroachment (50% of swing range)

For limit orders, scans forward within fill_timeout bars for fill.
All fill logic is fully causal (uses only data available at signal bar
plus forward scan within the timeout window).

Public API:
  compute_entry_price(df, i, entry_type, fill_timeout, direction)
    -> (price, filled_bar) or (NaN, -1)
"""

import numpy as np


def compute_entry_price(df, i, entry_type, fill_timeout, direction):
    """Compute entry price for a given signal bar and entry type.

    Parameters
    ----------
    df : DataFrame
        Must contain close, high, low, and relevant ICT columns.
    i : int
        Signal bar index.
    entry_type : str
        One of 'market', 'limit_ob_mid', 'limit_fvg_edge', 'limit_ce'.
    fill_timeout : int
        Max bars to wait for limit fill.
    direction : str
        'long' or 'short'.

    Returns
    -------
    (entry_price, filled_bar_index) : (float, int)
        Returns (NaN, -1) if no fill within timeout.
    """
    if entry_type == "market":
        return float(df["close"].iloc[i]), i

    n = len(df)
    limit_price = _get_limit_price(df, i, entry_type, direction)

    if np.isnan(limit_price):
        return np.nan, -1

    # Scan forward for fill within [i, i + fill_timeout]
    end = min(i + fill_timeout + 1, n)
    low_vals = df["low"].values
    high_vals = df["high"].values

    if direction == "long":
        # Buy limit: filled when low touches or goes below limit
        for j in range(i, end):
            if low_vals[j] <= limit_price:
                return limit_price, j
    else:
        # Sell limit (short entry): filled when high touches or goes above limit
        for j in range(i, end):
            if high_vals[j] >= limit_price:
                return limit_price, j

    return np.nan, -1


def _get_limit_price(df, i, entry_type, direction):
    """Compute limit price for a given entry type at bar i.

    Returns NaN if required columns have NaN at bar i.
    """
    if entry_type == "limit_ob_mid":
        return _limit_ob_mid(df, i, direction)
    elif entry_type == "limit_fvg_edge":
        return _limit_fvg_edge(df, i, direction)
    elif entry_type == "limit_ce":
        return _limit_ce(df, i)
    else:
        raise ValueError(f"Unknown entry_type: {entry_type}")


def _limit_ob_mid(df, i, direction):
    """Midpoint of most recent order block in trade direction."""
    if direction == "long":
        top = df["ict_ob_bull_top"].iloc[i]
        bot = df["ict_ob_bull_bot"].iloc[i]
    else:
        top = df["ict_ob_bear_top"].iloc[i]
        bot = df["ict_ob_bear_bot"].iloc[i]

    if np.isnan(top) or np.isnan(bot):
        return np.nan
    return (top + bot) / 2.0


def _limit_fvg_edge(df, i, direction):
    """Edge of nearest active FVG in trade direction.

    Long: lower edge of nearest bull FVG (buy dip into FVG).
    Short: upper edge of nearest bear FVG (sell rally into FVG).
    """
    if direction == "long":
        val = df["ict_fvg_bull_nearest_bot"].iloc[i]
    else:
        val = df["ict_fvg_bear_nearest_top"].iloc[i]

    return float(val) if not np.isnan(val) else np.nan


def _limit_ce(df, i):
    """Consequent encroachment: 50% of swing range.

    CE = (last_swing_high + last_swing_low) / 2.
    Uses forward-filled swing prices to ensure a value exists.
    """
    # Use ict_swing_high/low_price columns (forward-filled at call site
    # or here). These columns are sparse -- only populated at swing bars.
    # We need the last known swing prices.
    sh_col = df["ict_swing_high_price"]
    sl_col = df["ict_swing_low_price"]

    # Look backwards from bar i for last non-NaN values
    sh = _last_valid_before(sh_col.values, i)
    sl = _last_valid_before(sl_col.values, i)

    if np.isnan(sh) or np.isnan(sl):
        return np.nan
    return (sh + sl) / 2.0


def _last_valid_before(arr, i):
    """Find last non-NaN value at or before index i."""
    for j in range(i, -1, -1):
        if not np.isnan(arr[j]):
            return arr[j]
    return np.nan
