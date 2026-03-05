# core/engine/sizing.py
# Position sizing utilities. Ported from legacy/scripts/position_sizing.py.
# D38 -- Kelly fraction with per-trade ML probability.

from __future__ import annotations

import numpy as np


def kelly_fraction(
    p: float,
    odds: float = 2.0,
    divisor: float = 40.0,
) -> float:
    """
    Fractional Kelly bet size for a single trade probability.

    f = (1/divisor) * max(0, p - (1-p)/odds)

    With divisor=40, odds=2:
      p=0.60 -> 0.50%    p=0.70 -> 0.88%
      p=0.80 -> 1.25%    p=0.90 -> 1.63%

    Returns fraction clipped to [0.01, 0.02] (1% to 2% risk per trade).
    """
    raw = p - (1.0 - p) / odds
    raw = max(raw, 0.0)
    f = raw / divisor
    return max(0.01, min(f, 0.02))


def kelly_fraction_array(
    probs: np.ndarray,
    odds: float = 2.0,
    divisor: float = 40.0,
    floor: float = 0.0025,
    cap: float = 0.02,
) -> np.ndarray:
    """
    Vectorized fractional Kelly for an array of ML probabilities.

    f = (1/divisor) * max(0, p - (1-p)/odds), clipped to [floor, cap].
    """
    raw = probs - (1.0 - probs) / odds
    raw = np.maximum(raw, 0.0)
    f = raw / divisor
    return np.clip(f, floor, cap)


def equity_sim(
    r_returns: np.ndarray,
    risk_pct: float,
    initial_equity: float = 10_000.0,
) -> tuple[list[float], float]:
    """
    Compute equity path and max drawdown from R returns with fixed risk.
    Returns (equity_path, max_drawdown_fraction).
    """
    equity = initial_equity
    peak = equity
    max_dd = 0.0
    path = [equity]

    for r in r_returns:
        pnl = equity * risk_pct * r
        equity += pnl
        if equity <= 0:
            equity = 0.0
            path.append(equity)
            max_dd = 1.0
            break
        peak = max(peak, equity)
        dd = (peak - equity) / peak
        max_dd = max(max_dd, dd)
        path.append(equity)

    return path, max_dd


def equity_sim_variable(
    r_returns: np.ndarray,
    risk_pcts: np.ndarray,
    initial_equity: float = 10_000.0,
) -> tuple[list[float], float]:
    """
    Equity path with per-trade variable risk (from Kelly array).
    Returns (equity_path, max_drawdown_fraction).
    """
    equity = initial_equity
    peak = equity
    max_dd = 0.0
    path = [equity]

    for r, risk in zip(r_returns, risk_pcts):
        pnl = equity * risk * r
        equity += pnl
        if equity <= 0:
            equity = 0.0
            path.append(equity)
            max_dd = 1.0
            break
        peak = max(peak, equity)
        dd = (peak - equity) / peak
        max_dd = max(max_dd, dd)
        path.append(equity)

    return path, max_dd
